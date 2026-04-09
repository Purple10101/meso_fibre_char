"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260409

ws_connect_ex.py
ws connection example for client... but im not leaking my ip.
replace PUT_IPV4_HERE with the IPV4 of the systems host.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import asyncio
import websockets

async def listen():
    async with websockets.connect("ws://PUT_IPV4_HERE:8765") as ws:
        print("Connected")
        async for message in ws:
            print(f"Received: {message}")

asyncio.run(listen())