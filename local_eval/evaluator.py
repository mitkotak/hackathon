#!/usr/bin/env python3

import argparse
import time
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.client import PendingRequest
from client.example_model import NnInferenceClient


def score_fn(
    prediction,
    target,
    latency_ms,
    error_scale=1.0,
    latency_scale=100.0,
    error_exp=2.0,
    latency_exp=1.0,
):
    error = abs(prediction - target)
    error_term = 1 / (1 + (error * error_scale) ** error_exp)
    latency_term = 1 / (1 + (latency_ms / latency_scale) ** latency_exp)
    return error_term, latency_term, error_term * latency_term


class LocalEvaluator:
    def __init__(self, requests_file: str):
        self.requests_df = pd.read_parquet(requests_file)
        self.unique_symbols = self.requests_df["symbol"].unique()
        self.num_symbols = len(self.unique_symbols)

    def evaluate_model(
        self, process_batch_fn, batch_size: int = 32
    ) -> dict[str, float]:
        all_requests = []
        for idx, row in self.requests_df.iterrows():
            feature_cols = [f"feature_{i:02d}" for i in range(32)]
            features = [float(row[col]) for col in feature_cols]
            req = PendingRequest(
                uuid=row["uuid"],
                symbol=row["symbol"],
                features=features,
                received_time=time.time(),
            )
            all_requests.append(req)

        predictions = {}
        request_latencies = {}

        for i in range(0, len(all_requests), batch_size):
            batch = all_requests[i : i + batch_size]

            batch_by_symbol = {}
            for req in batch:
                if req.symbol not in batch_by_symbol:
                    batch_by_symbol[req.symbol] = []
                batch_by_symbol[req.symbol].append(req)

            start_time = time.perf_counter()
            responses = process_batch_fn(batch_by_symbol)
            end_time = time.perf_counter()

            batch_time_ms = (end_time - start_time) * 1000

            for uuid, pred in zip(responses.uuids, responses.predictions):
                predictions[uuid] = pred
                request_latencies[uuid] = batch_time_ms

        return self._calculate_metrics(predictions, request_latencies)

    def _calculate_metrics(self, predictions: dict, request_latencies: dict) -> dict:
        accuracy_scores = []
        latency_scores = []
        combined_scores = []
        latencies = []

        for idx, row in self.requests_df.iterrows():
            uuid = row["uuid"]
            target = row["target"]
            if uuid in predictions:
                prediction = predictions[uuid]
                latency_ms = request_latencies[uuid]

                latencies.append(latency_ms)

                acc_score, lat_score, combined_score = score_fn(
                    prediction, target, latency_ms
                )
                accuracy_scores.append(acc_score)
                latency_scores.append(lat_score)
                combined_scores.append(combined_score)

        response_rate = len(predictions) / len(self.requests_df)
        avg_latency = np.mean(latencies) if latencies else 0
        avg_accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 0
        avg_latency_score = np.mean(latency_scores) if latency_scores else 0
        avg_combined_score = np.mean(combined_scores) if combined_scores else 0

        return {
            "total_requests": len(self.requests_df),
            "total_responses": len(predictions),
            "response_rate": response_rate,
            "avg_latency_ms": avg_latency,
            "avg_accuracy_score": avg_accuracy_score,
            "avg_latency_score": avg_latency_score,
            "overall_score": avg_combined_score,
        }

    def print_report(self, metrics: Dict[str, float]):
        print(f"\nTotal requests: {metrics['total_requests']:,}")
        print(f"Total responses: {metrics['total_responses']:,}")
        print(f"Response rate: {metrics['response_rate']:.2%}")
        print(f"Average accuracy score: {metrics['avg_accuracy_score']:.4f}")
        print(f"Average latency score: {metrics['avg_latency_score']:.4f}")
        print(f"Average latency: {metrics['avg_latency_ms']:.1f} ms")
        print(f"Overall score: {metrics['overall_score']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluator = LocalEvaluator(args.requests)
    client = NnInferenceClient(num_symbols=evaluator.num_symbols)

    # Override the client's states to match actual symbols in data
    client.states = {
        symbol: client.model.init_state(1, client.device)
        for symbol in evaluator.unique_symbols
    }

    metrics = evaluator.evaluate_model(client.process_batch, batch_size=args.batch_size)
    evaluator.print_report(metrics)


if __name__ == "__main__":
    main()
