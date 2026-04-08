"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260408

websocket.py
reconstruction of fragmented fibres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import json
import websockets

connected_clients = set()


async def handler(websocket):
    connected_clients.add(websocket)
    print(f"Client connected ({len(connected_clients)} total)")
    try:
        async for message in websocket:
            # Future: handle client commands here
            print(f"Received: {message}")
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected ({len(connected_clients)} total)")


async def broadcast(data):
    for ws in connected_clients.copy():
        try:
            await ws.send(json.dumps(data))
        except websockets.ConnectionClosed:
            connected_clients.discard(ws)


async def start_server(host="0.0.0.0", port=8765):
    await websockets.serve(handler, host, port)
    print(f"WebSocket server running on ws://{host}:{port}")