import socket
import json
import base64
import cv2
import numpy as np

def receive_full_data(sock):
    data = b''
    while True:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet

        if b'\n' in data:
            break

    return data.split(b'\n')

def main(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print(f'Connected to server at {host}:{port}')

        camera_work_id = "camera.1000"
        camera_json_data = {
            "request_id": "001",
            "work_id": "camera",
            "action": "setup",
            "object": "camera.setup",
            "data": {
                "response_format":"camera.raw",
                "input":"/dev/video0",
                "enoutput":False,
                "frame_width":384,
                "frame_height":256
            }
        }
        json_string = json.dumps(camera_json_data)
        client_socket.sendall(f"{len(json_string):<10}".encode('utf-8'))
        client_socket.sendall(json_string.encode('utf-8'))

        while True:
            data_packets = receive_full_data(client_socket)
            if not data_packets:
                print("No more data from server, closing connection...")
                break

            for data in data_packets:
                if not data:
                    continue

                try:
                    print(f'Received data: {data.decode("utf-8")}')
                    json_data = json.loads(data.decode('utf-8'))
                    camera_work_id = json_data['work_id']
                    break

                except json.JSONDecodeError as e:
                    print(f'Error parsing JSON: {e}')
                except KeyError as e:
                    print(f'Key error: {e}')
                except Exception as e:
                    print(f'Unexpected error: {e}')

            if 'camera_work_id' in locals():
                break

        depth_anything_json_data = {
            "request_id": "002",
            "work_id": "depth_anything",
            "action": "setup",
            "object": "depth_anything.setup",
            "data": {
                "model": "depth-anything-ax630c",
                "response_format": "jpeg.base64.stream",
                "input": camera_work_id,
                "enoutput": True
            }
        }

        json_string = json.dumps(depth_anything_json_data)
        client_socket.sendall(f"{len(json_string):<10}".encode('utf-8'))
        client_socket.sendall(json_string.encode('utf-8'))

        while True:
            data_packets = receive_full_data(client_socket)
            if not data_packets:
                print("No more data from server, closing connection...")
                break

            for data in data_packets:
                if not data:
                    continue

                try:
                    # print(f'Received data: {data.decode("utf-8")}')
                    json_data = json.loads(data.decode('utf-8'))
                    delta = json_data['data']['delta']

                    img_data = base64.b64decode(delta)
                    nparr = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    cv2.imshow('Depth_Anything Video', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting...")
                        break

                except json.JSONDecodeError as e:
                    print(f'Error parsing JSON: {e}')
                except KeyError as e:
                    print(f'Key error: {e}')
                except Exception as e:
                    print(f'Unexpected error: {e}')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    finally:
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    host = "192.168.20.104"
    port = 10001
    main(host, port)