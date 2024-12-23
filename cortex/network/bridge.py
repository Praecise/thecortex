# cortex/network/bridge.py

import asyncio
import websockets
import json
from typing import Optional, Dict, Any
from .protocol import TenzroProtocol, MessageType

class TenzroBridge:
    """Bridge between Cortex and Tenzro network"""
    
    def __init__(
        self,
        websocket_url: str,
        node_id: str,
        node_type: str,
        region: str,
        protocol: Optional[TenzroProtocol] = None
    ):
        self.ws_url = websocket_url
        self.node_id = node_id
        self.node_type = node_type
        self.region = region
        self.protocol = protocol or TenzroProtocol(node_id, node_type, region)
        self.ws = None
        self.connected = False
        self._running = True
        
    async def connect(self) -> None:
        """Connect to Tenzro network"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.connected = True
            
            # Send join message
            join_message = await self.protocol.send_message(
                MessageType.JOIN,
                {
                    "resources": await self.protocol._get_resource_metrics()
                }
            )
            await self.ws.send(json.dumps(join_message))
            
            # Start message handling loop
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Failed to connect: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from network"""
        self._running = False
        if self.ws:
            try:
                leave_message = await self.protocol.send_message(
                    MessageType.LEAVE,
                    {}
                )
                await self.ws.send(json.dumps(leave_message))
            except:
                pass
            finally:
                await self.ws.close()
                self.connected = False
    
    async def _handle_messages(self) -> None:
        """Handle incoming messages"""
        while self._running and self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                await self.protocol.handle_message(data)
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                break
            except Exception as e:
                print(f"Error handling message: {str(e)}")
                continue
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message through websocket"""
        if self.connected and self.ws:
            await self.ws.send(json.dumps(message))