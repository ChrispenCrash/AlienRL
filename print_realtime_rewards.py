import socket
import pickle
from time import sleep
import os
import keyboard

def receive_data(host, port):

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((host, port))
        s.settimeout(1)
        while True:
            try:
                data, addr = s.recvfrom(1024)
            except socket.timeout:
                continue
            if not data:
                break

            message = pickle.loads(data)

            progress_reward = message['progress_reward']
            on_track_reward = message['on_track_reward']
            car_damage_penalty = message['car_damage_penalty']
            orientation_reward = message['orientation_reward']

            total_reward = progress_reward + on_track_reward + car_damage_penalty + orientation_reward

            print("Progress | On Track |  Damage |  Angle |  Total")
            print(f"{progress_reward:6.2f}   | {on_track_reward:6.1f}   | {car_damage_penalty:7.1f} | {orientation_reward:6.2f} | {total_reward:6.1f}")
            
            sleep(0.1)
            os.system("cls")

            if keyboard.is_pressed("q"):
                print("Stopping.")
                break

if __name__ == "__main__":
    os.system('cls')
    print("Listening for input...")
    receive_data('localhost', 12345)