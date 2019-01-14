import asyncio
import websockets
import argparse
import json

async def send_query(text,server,port):
    async with websockets.connect('ws://{}:{}'.format(server,port)) as websocket:
        await websocket.send(text)
        result = await websocket.recv()
        return result

def main(args):
    text = input("Query: ")
    while text != "quit":
        result = asyncio.get_event_loop().run_until_complete(send_query(text,args.hostname,args.port))
        result = json.loads(result)
        print_result(result)
        text = input("Query: ")

def print_result(result):
    return json.dumps(result,ensure_ascii=False,indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Send queries to a websocket server.")
    parser.add_argument("--hostname", type=str, default="localhost", help="hostname of server")
    parser.add_argument("--port", type=int, default=8765, help="port of server")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
