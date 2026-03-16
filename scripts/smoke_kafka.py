#!/usr/bin/env python3
"""Smoke test: verify Kafka connectivity, topic existence, and produce/consume round-trip.

Usage:
    python scripts/smoke_kafka.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time


async def main() -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.admin import AIOKafkaAdminClient

    from matyan_backend.config import SETTINGS

    bootstrap = SETTINGS.kafka_bootstrap_servers
    data_topic = SETTINGS.kafka_data_ingestion_topic
    control_topic = SETTINGS.kafka_control_events_topic

    print(f"Kafka bootstrap: {bootstrap}")
    print(f"Topics: {data_topic}, {control_topic}")
    print()

    # --- 1. Check topics exist ---
    print("1. Checking topics via admin client...")
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        topics = await admin.list_topics()
        for t in (data_topic, control_topic):
            if t in topics:
                print(f"   [OK] Topic '{t}' exists")
            else:
                print(f"   [FAIL] Topic '{t}' NOT found (available: {topics})")
                sys.exit(1)
    finally:
        await admin.close()

    # --- 2. Produce a test message ---
    print("\n2. Producing test message...")
    producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    await producer.start()
    test_payload = {"_smoke_test": True, "ts": time.time()}
    try:
        record = await producer.send_and_wait(data_topic, value=test_payload)
        print(f"   [OK] Sent to {record.topic} partition={record.partition} offset={record.offset}")
    finally:
        await producer.stop()

    # --- 3. Consume the message back ---
    print("\n3. Consuming test message...")
    consumer = AIOKafkaConsumer(
        data_topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset="earliest",
        group_id="smoke-test",
        value_deserializer=json.loads,
        consumer_timeout_ms=10000,
    )
    await consumer.start()
    found = False
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            batch = await consumer.getmany(timeout_ms=2000)
            for messages in batch.values():
                for msg in messages:
                    if isinstance(msg.value, dict) and msg.value.get("_smoke_test"):
                        print(f"   [OK] Received: {msg.value}")
                        found = True
                        break
                if found:
                    break
            if found:
                break
    finally:
        await consumer.stop()

    if not found:
        print("   [FAIL] Did not receive test message within timeout")
        sys.exit(1)

    print("\nAll Kafka smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
