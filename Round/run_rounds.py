#!/usr/bin/env python3
"""
SGLang Multi-Turn Conversation Benchmark (Optimized)

优化点：
1. 修正 Cache 命中率统计：使用增量 token 计算真实本轮命中率。
2. 增强多轮对话鲁棒性：部分失败时继续运行，使用首个成功回复。
3. 支持精确 Tokenizer 计数 (--tokenizer)。
4. 使用 Session 复用连接，提升并发性能。
5. 预生成 Dummy 文本池，减少 CPU 开销。
6. TTFT 测量点改为收到第一个 SSE 帧，而非首个 content token。
7. 支持 --record-response 开关记录流式返回的对话内容，每轮增量保存，Ctrl+C 中断时保存已有数据。
8. 支持 --increment-tokens 0 时每轮发送相同请求，不累积历史。
9. 使用 io.StringIO 替代字符串拼接缓存流式响应，提升性能。

注意：获取每请求 kvcache 命中率依赖服务端返回 usage.prompt_tokens_details.cached_tokens，
需加 --request-usage 并在启动 sglang 服务时加 --enable-cache-report；否则退回用全局指标增量估算。
PD 分离（经 router）部署时，若 --request-usage 导致请求失败，请关闭该开关（默认关闭）。
"""

import argparse
import time
import json
import statistics
import requests
import random
import string
import re
import io
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class SingleRequestResult:
    success: bool
    response: Optional[str] = None
    ttft_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    error: Optional[str] = None
    output_tokens: int = 0
    prompt_tokens: Optional[int] = None  # 本次请求输入 prompt token 数（来自 usage.prompt_tokens）
    cached_tokens: Optional[int] = None  # 本次请求 kvcache 命中 token 数（来自 usage.prompt_tokens_details.cached_tokens）
    raw_chunks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoundBatchResult:
    round_num: int
    total_input_tokens: int
    batch_size: int
    successful_requests: int
    failed_requests: int
    avg_ttft_ms: float
    max_ttft_ms: float
    min_ttft_ms: float
    avg_total_latency_ms: float
    total_time_ms: float
    throughput_rps: float
    cache_hit_rate: float
    cached_tokens_delta: int
    total_prompt_tokens_delta: int


class SGLangBatchBenchmark:
    def __init__(self, base_url: str, model: str, tokenizer=None, api_key: Optional[str] = None,
                 record_response: bool = False, use_fix_prompt: bool = False, seed: int = 20262223,
                 request_usage: bool = False):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.tokenizer = tokenizer
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self._dummy_pool = None
        self._dummy_pool_len = 0
        self._text_counter = 0  # 用于生成不同的文本
        self.record_response = record_response
        self.use_fix_prompt = use_fix_prompt
        self.seed = seed
        self.request_usage = request_usage  # 是否请求 usage chunk（需服务端支持 stream_options.include_usage）
        # v6: 语料库相关属性
        self._corpus_tokens = None
        self._corpus_tokens_len = 0
        self._corpus_file = "./doupocangqiong.txt"
        self._tokenizer = tokenizer  # 保存 tokenizer 引用

    def _init_dummy_pool(self, max_tokens: int):
        """v6: 从语料库文件加载并 tokenize"""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            except ImportError:
                raise RuntimeError("transformers not installed, cannot use corpus mode")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {e}")

        with open(self._corpus_file, "r", encoding="utf-8") as f:
            self._corpus_tokens = self._tokenizer.encode(f.read(), skip_special_tokens=True)
        self._corpus_tokens_len = len(self._corpus_tokens)
        print(f"[v6] Corpus loaded: {self._corpus_tokens_len} tokens from {self._corpus_file}")

    def _generate_dummy_text(self, token_len: int, unique_id: int = 0) -> str:
        """v6: 从语料库中根据种子截取指定 token 数"""
        if self._corpus_tokens is None:
            self._init_dummy_pool(max(token_len, 16000))

        if token_len > self._corpus_tokens_len:
            raise ValueError(f"Requested {token_len} tokens, but corpus only has {self._corpus_tokens_len} tokens")

        # 使用 unique_id + counter 确保每个请求的文本都不同
        seed_offset = (self._text_counter + unique_id) * 7919  # 7919 是质数，减少冲突

        if self.use_fix_prompt:
            random.seed(self.seed + seed_offset)
        else:
            random.seed()  # 重置随机种子

        max_start = self._corpus_tokens_len - token_len
        if max_start <= 0:
            start = 0
        else:
            start = random.randint(0, max_start)

        self._text_counter += 1
        # 精确截取指定 token 数并解码

        res = self._tokenizer.decode(self._corpus_tokens[start:start + token_len], skip_special_tokens=True)
        print(f"[PROMPT] requested {token_len} tokens, text len: {len(res)}")

        return res

    def count_tokens(self, text: str, kind: str = "INPUT") -> int:
        if self._tokenizer:
            token_len = len(self._tokenizer.encode(text))
        else:
            token_len = len(text) // 4
        has_tokenizer = True if self._tokenizer else False
        print(f"[{kind}] {token_len} tokens ({'exact' if has_tokenizer else 'approx'})")

        return token_len

    def get_server_metrics(self) -> Dict[str, float]:
        # PD 分离经 router 时，router 的指标端点是 /engine_metrics（聚合各 worker 的 /metrics，
        # 且把指标名中的 : 替换为 _）；直连 SRT 时用 /metrics。两种都试。
        for path in ("/engine_metrics", "/metrics"):
            try:
                resp = self.session.get(f"{self.base_url}{path}", timeout=5)
                if resp.status_code == 200:
                    parsed = self._parse_metrics(resp.text)
                    if parsed:
                        return parsed
            except Exception:
                pass
        return {}

    def _parse_metrics(self, text: str) -> Dict[str, float]:
        metrics = {}
        # 注意：
        # 1. sglang SRT 指标名是 sglang:*（冒号）；经 router /engine_metrics 聚合后冒号被替换为下划线 sglang_*，
        #    且同一指标可能有多条（不同 source/worker）。这里取所有匹配行的最大值，避免把多个 worker 的
        #    计数误当作单值（聚合后多个 worker 的 counter 不会自动相加，需累加才准；max 是保守近似）。
        # 2. 累计计数器名是 *_total 后缀，cache_hit_rate 是 gauge（瞬时值）。
        patterns = {
            'cache_hit_rate': r'sglang[:_]cache_hit_rate\s+([\d.]+)',
            'cached_tokens': r'sglang[:_]cached_tokens_total\s+([\d.eE+]+)',
            'prompt_tokens': r'sglang[:_]prompt_tokens_total\s+([\d.eE+]+)',
        }
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # 多条（多 worker）时取和更接近总量；单条时直接取值
                    metrics[key] = sum(float(m) for m in matches)
                except ValueError:
                    pass
        return metrics

    def send_chat_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 32,
        temperature: float = 0.0,
        routing_key: Optional[str] = None,
    ) -> SingleRequestResult:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        # 可选：请求末尾的 usage chunk（含 prompt_tokens_details.cached_tokens），
        # 用于获取每请求的 kvcache 命中数。需服务端 --enable-cache-report 才会填充 cached_tokens。
        # 默认关闭，避免不兼容 stream_options 的网关/路由拒绝请求（PD 分离走 router 时尤其注意）。
        if self.request_usage:
            payload["stream_options"] = {"include_usage": True}

        # Attach routing-key header for DP affinity in multi-turn PD
        headers = {}
        if routing_key is not None:
            headers["x-smg-routing-key"] = routing_key

        start_time = time.perf_counter()
        first_token_time = None
        response_buffer = io.StringIO()
        raw_chunks: List[Dict[str, Any]] = []
        usage_cached_tokens = None
        usage_prompt_tokens = None
        error = None
        try:
            # """

            with self.session.post(url, json=payload, headers=headers, stream=True, timeout=300) as resp:
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return SingleRequestResult(success=False, error=error)

                # ---------- 修改点：TTFT 记录在收到第一个 SSE 帧时 ----------
                first_frame_received = False
                for line in resp.iter_lines():
                    if not line:
                        continue
                    # 只要收到非空行，立即记录 TTFT（无论内容是什么）
                    if not first_frame_received:
                        first_token_time = time.perf_counter()
                        first_frame_received = True

                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if self.record_response:
                                raw_chunks.append(chunk)
                            if 'error' in chunk:
                                error = chunk['error'].get('message', 'Unknown API error')
                                return SingleRequestResult(success=False, error=error)
                            # usage chunk（choices 为空数组）携带每请求的 token 统计
                            usage = chunk.get('usage')
                            if usage:
                                usage_prompt_tokens = usage.get('prompt_tokens')
                                details = usage.get('prompt_tokens_details') or {}
                                usage_cached_tokens = details.get('cached_tokens', 0)
                            # 注意：usage chunk 的 choices 为空数组，直接 [0] 会越界
                            choices = chunk.get('choices')
                            if choices:
                                delta = choices[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    response_buffer.write(content)
                        except json.JSONDecodeError as e:
                            # 打印解析失败的数据行（可能是 usage chunk 被分帧截断），便于诊断
                            print(f"    [warn] JSON parse failed: {e} | line={data_str[:120]!r}")
                            continue
            # """
            # full_response = "diy response !!!"

            full_response = response_buffer.getvalue()
            total_time = time.perf_counter() - start_time
            ttft = (first_token_time - start_time) * 1000 if first_token_time else None
            output_tokens = self.count_tokens(full_response, kind="OUTPUT") if full_response else 0

            return SingleRequestResult(
                success=True,
                response=full_response,
                ttft_ms=ttft,
                total_latency_ms=total_time * 1000,
                output_tokens=output_tokens,
                prompt_tokens=usage_prompt_tokens,
                cached_tokens=usage_cached_tokens if usage_prompt_tokens is not None else None,
                raw_chunks=raw_chunks,
            )

        except Exception as e:
            return SingleRequestResult(success=False, error=str(e))

    def run_batch_round(
        self,
        messages_list: List[List[Dict[str, str]]],
        batch_size: int,
        max_workers: Optional[int] = None,
        output_length: int = 32,
        temperature: float = 0.0,
        routing_keys: Optional[List[Optional[str]]] = None,
    ) -> Tuple[RoundBatchResult, List[Tuple[int, SingleRequestResult]]]:
        """运行一轮批量请求，每个请求有独立的对话历史

        Args:
            messages_list: 每个请求的独立对话历史列表
            batch_size: 批量大小 (可以小于 len(messages_list))
            max_workers: 最大并发数
            output_length: 输出 token 数
            temperature: 采样温度
            routing_keys: 每个请求的 routing key，用于 DP 亲和

        Returns:
            batch_result: 本轮统计结果
            results: List of (original_idx, SingleRequestResult) tuples
        """
        workers = max_workers or batch_size
        results: List[Tuple[int, SingleRequestResult]] = []
        start_time = time.perf_counter()

        pre_metrics = self.get_server_metrics()
        pre_cached = pre_metrics.get('cached_tokens', 0)
        pre_prompt = pre_metrics.get('prompt_tokens', 0)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx, messages in enumerate(messages_list[:batch_size]):
                rk = routing_keys[idx] if routing_keys and idx < len(routing_keys) else None
                futures.append((
                    idx,
                    executor.submit(
                        self.send_chat_request,
                        messages,
                        output_length,
                        temperature,
                        rk,
                    )
                ))

            for idx, future in futures:
                try:
                    res = future.result()
                    results.append((idx, res))  # 保存原始 idx 以便正确更新对话历史
                except Exception as e:
                    results.append((idx, SingleRequestResult(success=False, error=str(e))))

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        post_metrics = self.get_server_metrics()
        post_cached = post_metrics.get('cached_tokens', 0)
        post_prompt = post_metrics.get('prompt_tokens', 0)

        delta_cached = max(0, post_cached - pre_cached)
        delta_prompt = max(0, post_prompt - pre_prompt)
        server_hit_rate = (delta_cached / delta_prompt) if delta_prompt > 0 else 0.0

        successful = [r for _, r in results if r.success]
        failed = [r for _, r in results if not r.success]
        success_count = len(successful)
        failed_count = len(failed)

        # 临时诊断：打印每请求 usage 原始值，用于区分「未拿到 usage」与「真的未命中」
        for idx, r in results:
            if r.success:
                print(f"\n[Diag Req {idx}] [User {idx}] prompt_tokens={r.prompt_tokens}, cached_tokens={r.cached_tokens}")
            else:
                print(f"\n[Diag Req {idx}] [User {idx}] FAILED: {r.error}")

        # 优先使用每请求 usage 中的 cached_tokens（精确、按轮归属）：
        # 命中率 = Σcached / Σprompt。仅当服务端未开 --enable-cache-report 时
        # cached_tokens 为 None，此时退回用全局计数器 delta 估算。
        per_req_cache = [
            (r.cached_tokens, r.prompt_tokens)
            for r in successful
            if r.cached_tokens is not None and r.prompt_tokens is not None
        ]
        if per_req_cache:
            sum_cached = sum(c for c, _ in per_req_cache)
            sum_prompt = sum(p for _, p in per_req_cache)
            real_cache_hit_rate = (sum_cached / sum_prompt) if sum_prompt > 0 else 0.0
            delta_cached = int(sum_cached)
            delta_prompt = int(sum_prompt)
        else:
            real_cache_hit_rate = server_hit_rate

        if success_count > 0:
            ttfts = [r.ttft_ms for _, r in results if r.success and r.ttft_ms is not None]
            latencies = [r.total_latency_ms for _, r in results if r.success and r.total_latency_ms is not None]
            avg_ttft = statistics.mean(ttfts) if ttfts else 0
            max_ttft = max(ttfts) if ttfts else 0
            min_ttft = min(ttfts) if ttfts else 0
            avg_lat = statistics.mean(latencies) if latencies else 0
        else:
            avg_ttft = max_ttft = min_ttft = avg_lat = 0

        throughput = success_count / (total_time_ms / 1000) if total_time_ms > 0 else 0

        batch_result = RoundBatchResult(
            round_num=0,
            total_input_tokens=0,
            batch_size=batch_size,
            successful_requests=success_count,
            failed_requests=failed_count,
            avg_ttft_ms=avg_ttft,
            max_ttft_ms=max_ttft,
            min_ttft_ms=min_ttft,
            avg_total_latency_ms=avg_lat,
            total_time_ms=total_time_ms,
            throughput_rps=throughput,
            cache_hit_rate=real_cache_hit_rate,
            cached_tokens_delta=int(delta_cached),
            total_prompt_tokens_delta=int(delta_prompt),
        )
        return batch_result, results

    def run_conversation(
        self,
        initial_tokens: int,
        increment_tokens: int,
        rounds: int,
        batch_size: int = 1,
        max_workers: Optional[int] = None,
        output_length: int = 32,
        temperature: float = 0.0,
    ) -> Tuple[List[RoundBatchResult], List[List[Dict[str, str]]]]:
        """运行多轮对话测试，每个请求维护独立的对话历史

        每个请求在每轮都会收到不同的 prompt，但都会继承自己之前的历史

        Returns:
            results: 每轮的统计结果列表
            all_messages: 每个用户的完整对话历史列表
        """
        max_needed = max(initial_tokens, increment_tokens)
        max_needed = initial_tokens + increment_tokens * batch_size
        self._init_dummy_pool(max_needed * (rounds + 1))

        results = []
        # 每个请求独立的对话历史列表（用于发送请求）
        all_messages = [[] for _ in range(batch_size)]
        # 独立的记录列表，累积所有轮次的 user+assistant（用于 record_response）
        recorded_messages = [[] for _ in range(batch_size)]
        total_input_tokens_list = [0] * batch_size
        # 每个对话一个固定的 routing_key，保证同一对话所有轮次路由到同一 DP rank
        routing_keys = [f"{idx}" for idx in range(batch_size)]
        current_user_len = initial_tokens

        no_increment = (increment_tokens == 0)

        if no_increment:
            fixed_user_contents = []
            for idx in range(batch_size):
                user_content = self._generate_dummy_text(initial_tokens, unique_id=idx)
                fixed_user_contents.append("summarise the flowing text:" + user_content)
                total_input_tokens_list[idx] += self.count_tokens(user_content, kind="INPUT")

        for round_num in range(1, rounds + 1):
            # 为每个请求生成不同的 user message
            user_contents = []
            if no_increment:
                for idx in range(batch_size):
                    all_messages[idx] = [{"role": "user", "content": fixed_user_contents[idx]}]
                    # 记录每轮的 user 消息
                    recorded_messages[idx].append({"role": "user", "content": fixed_user_contents[idx]})
            else:
                for idx in range(batch_size):
                    user_content = self._generate_dummy_text(current_user_len, unique_id=idx)
                    user_contents.append("summarise the flowing:" + user_content)
                    all_messages[idx].append({"role": "user", "content": user_content})
                    recorded_messages[idx].append({"role": "user", "content": user_content})
                    total_input_tokens_list[idx] += self.count_tokens(user_content, kind="INPUT")

            batch_result, individual_results = self.run_batch_round(
                messages_list=all_messages,
                batch_size=batch_size,
                max_workers=max_workers,
                output_length=output_length,
                temperature=temperature,
                routing_keys=routing_keys,
            )
            time.sleep(2)
            batch_result.round_num = round_num
            # 每轮单个请求的累积输入 token 数，与 batch_size 无关
            batch_result.total_input_tokens = int(total_input_tokens_list[0])
            results.append(batch_result)

            # 更新每个请求的对话历史
            successful = [r for _, r in individual_results if r.success]
            if successful:
                if no_increment:
                    # no_increment 模式下不累积到 all_messages（下轮会重置），
                    # 但记录 assistant 回复到 recorded_messages
                    for idx, res in individual_results:
                        if res.success and res.response:
                            recorded_messages[idx].append({"role": "assistant", "content": res.response})
                else:
                    for idx, res in individual_results:
                        if res.success and res.response:
                            all_messages[idx].append({"role": "assistant", "content": res.response})
                            recorded_messages[idx].append({"role": "assistant", "content": res.response})
                            total_input_tokens_list[idx] += res.output_tokens
                total_input_tokens_dis = sum(total_input_tokens_list)
                print(f"[Round {round_num:2d}] Success. Total context ~{int(total_input_tokens_dis)} tokens.")
            else:
                print(f"[Round {round_num:2d}] CRITICAL: All requests failed. Stopping test.")
                for idx, res in individual_results:
                    print(f"    [req {idx}] error: {res.error}")
                break

            self._print_batch_result(batch_result)
            current_user_len = increment_tokens

            if self.record_response:
                global _shutdown_messages
                _shutdown_messages = [list(msgs) for msgs in recorded_messages]
                save_conversations(recorded_messages, _shutdown_conversation_dir)

            time.sleep(0.5)

            pause = input("等待清洗hbm")

        # 返回 recorded_messages 以便 main 中保存完整对话
        return results, recorded_messages

    def _print_batch_result(self, result: RoundBatchResult):
        status = f"{result.successful_requests}/{result.batch_size} succ"
        print(f"[Round {result.round_num:2d}] {status} | "
              f"Total Input: {result.total_input_tokens:6d} tok | "
              f"Avg TTFT: {result.avg_ttft_ms:8.2f} ms | "
              f"Max TTFT: {result.max_ttft_ms:8.2f} ms | "
              f"Throughput: {result.throughput_rps:6.2f} req/s | "
              f"Cache Hit: {result.cache_hit_rate*100:5.1f}% | "
              f"Cached: {result.cached_tokens_delta:6d}/{result.total_prompt_tokens_delta:6d} tok")


def save_conversations(all_messages: List[List[Dict[str, str]]], conversation_dir: str):
    """将所有用户的对话历史保存到文件"""
    if not all_messages:
        return
    os.makedirs(conversation_dir, exist_ok=True)
    for user_idx, messages in enumerate(all_messages):
        filepath = os.path.join(conversation_dir, f"user_{user_idx}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"用户 {user_idx} 对话记录\n")
            f.write("=" * 60 + "\n\n")
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    f.write(f"[User]\n")
                    f.write(content)
                    f.write("\n\n")
                elif role == "assistant":
                    f.write(f"[Assistant]\n")
                    f.write(content)
                    f.write("\n\n")
                f.write("-" * 60 + "\n\n")
        print(f"对话记录已保存: {filepath}")


# 全局变量，用于信号处理时保存已有数据
_shutdown_messages: List[List[Dict[str, str]]] = []
_shutdown_conversation_dir = "conversations"
_shutdown_save_enabled = False


def _signal_handler(signum, frame):
    print("\n\n收到中断信号，正在保存已有对话记录...")
    if _shutdown_save_enabled and _shutdown_messages:
        save_conversations(_shutdown_messages, _shutdown_conversation_dir)
    print("保存完成，退出。")
    exit(0)


def main():
    parser = argparse.ArgumentParser(description="SGLang Multi-Turn Benchmark (Optimized)")
    parser.add_argument("--base-url", default="http://localhost:30000", help="SGLang server base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name for exact counting (optional)")
    parser.add_argument("--rounds", type=int, default=10, help="Number of conversation rounds")
    parser.add_argument("--initial-tokens", type=int, default=16, help="Token length of the first user message")
    parser.add_argument("--increment-tokens", type=int, default=16000, help="Token length of each subsequent new user message")
    parser.add_argument("--output-len", type=int, default=32, help="Number of tokens to generate per response")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of concurrent requests per round")
    parser.add_argument("--max-workers", type=int, default=None, help="Max threads for concurrency (default: batch-size)")
    parser.add_argument("--api-key", help="OpenAI API key (if required)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output-json", type=str, default=None, help="Save detailed results to JSON file")
    parser.add_argument("--record-response", action="store_true", help="记录流式返回的对话内容，测试结束后保存到文件")
    parser.add_argument("--conversation-dir", type=str, default="conversations", help="对话记录保存目录 (默认: conversations)")
    parser.add_argument("--use-fix-prompt", action="store_true",
                        help="使用固定 prompt（基于 SEED 生成），保证各次运行的 prompt 与总长度一致")
    parser.add_argument("--seed", type=int, default=20260915 + 1308,
                        help="prompt 生成随机种子（默认: 20260915+1308=20262223）")
    parser.add_argument("--request-usage", action="store_true",
                        help="请求 usage chunk 以获取每请求 kvcache 命中数（需服务端 --enable-cache-report 且支持 stream_options；PD 分离走 router 时若报错请关闭此开关）")

    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"Loaded tokenizer: {args.tokenizer}")
        except ImportError:
            print("Warning: transformers not installed, falling back to approximate token counting.")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer ({e}), falling back to approximate counting.")

    print("=" * 80)
    print("SGLang Multi-Turn Benchmark (Optimized)")
    print("=" * 80)
    print(f"Server: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer or 'Approximate (len//4)'}")
    print(f"Rounds: {args.rounds}")
    print(f"Initial tokens: {args.initial_tokens} | Increment: {args.increment_tokens}")
    print(f"Output len: {args.output_len} | Batch: {args.batch_size} | Workers: {args.max_workers or args.batch_size}")
    print(f"Record Response: {args.record_response}")
    print(f"Request Usage (per-req cached_tokens): {args.request_usage}")
    print("=" * 80)

    if args.use_fix_prompt:
        random.seed(args.seed)

    bench = SGLangBatchBenchmark(
        args.base_url, args.model, tokenizer, args.api_key,
        record_response=args.record_response,
        use_fix_prompt=args.use_fix_prompt,
        seed=args.seed,
        request_usage=args.request_usage,
    )

    global _shutdown_messages, _shutdown_conversation_dir, _shutdown_save_enabled
    if args.record_response:
        _shutdown_conversation_dir = args.conversation_dir
        _shutdown_save_enabled = True
        signal.signal(signal.SIGINT, _signal_handler)

    start_total = time.perf_counter()
    results, all_messages = bench.run_conversation(
        initial_tokens=args.initial_tokens,
        increment_tokens=args.increment_tokens,
        rounds=args.rounds,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        output_length=args.output_len,
        temperature=args.temperature,
    )
    end_total = time.perf_counter()

    print("\n" + "=" * 80)
    print("Summary (Using Incremental Cache Hit Rate)")
    print("=" * 80)

    successful_rounds = [r for r in results if r.successful_requests > 0]
    if successful_rounds:
        avg_ttft = statistics.mean(r.avg_ttft_ms for r in successful_rounds)
        avg_throughput = statistics.mean(r.throughput_rps for r in successful_rounds)
        avg_cache = statistics.mean(r.cache_hit_rate for r in successful_rounds)
        total_success = sum(r.successful_requests for r in results)
        total_fail = sum(r.failed_requests for r in results)
        print(f"Total Time: {end_total - start_total:.2f} s")
        print(f"Average TTFT (across rounds): {avg_ttft:.2f} ms")
        print(f"Average Throughput: {avg_throughput:.2f} req/s")
        print(f"Average Cache Hit Rate (real per-round): {avg_cache*100:.1f}%")
        print(f"Total successful requests: {total_success}")
        print(f"Total failed requests: {total_fail}")
    else:
        print("No successful requests!")
        print("  (请查看上方 CRITICAL 分支打印的具体错误)")

    print("\nDetailed Results (per round):")
    print("-" * 100)
    print(f"{'Round':>6} | {'Total In':>12} | {'Succ/Total':>10} | {'Avg TTFT':>10} | {'Max TTFT':>10} | {'Throughput':>10} | {'Cache Hit':>10} | {'Cached/Prompt':>15}")
    print("-" * 100)
    for r in results:
        print(f"{r.round_num:6d} | {r.total_input_tokens:12d} | {r.successful_requests:3d}/{r.batch_size:3d}    | "
              f"{r.avg_ttft_ms:10.2f} | {r.max_ttft_ms:10.2f} | {r.throughput_rps:10.2f} | "
              f"{r.cache_hit_rate*100:9.1f}% | {r.cached_tokens_delta:6d}/{r.total_prompt_tokens_delta:6d}")
    print("-" * 100)

    if args.output_json:
        output_data = {
            "config": vars(args),
            "results": [
                {
                    "round": r.round_num,
                    "total_input_tokens": r.total_input_tokens,
                    "avg_ttft_ms": r.avg_ttft_ms,
                    "max_ttft_ms": r.max_ttft_ms,
                    "throughput_rps": r.throughput_rps,
                    "cache_hit_rate": r.cache_hit_rate,
                    "cached_tokens_delta": r.cached_tokens_delta,
                    "total_prompt_tokens_delta": r.total_prompt_tokens_delta,
                }
                for r in results
            ]
        }
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    if args.record_response and all_messages:
        save_conversations(all_messages, args.conversation_dir)


if __name__ == "__main__":
    main()
