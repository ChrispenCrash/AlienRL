import socket
import pickle
from time import sleep
import os
import keyboard

def receive_data(host, port):

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((host, port))
        # s.settimeout(1)
        while True:
            # try:
            data, addr = s.recvfrom(1024)
            # except socket.timeout:
            #     continue
            # if not data:
            #     break

            message = pickle.loads(data)

            progress_reward = message['progress_reward']
            on_track_reward = message['on_track_reward'] # not updating correctly for some reason
            car_damage_penalty = message['car_damage_penalty']
            orientation_reward = message['orientation_reward']
            slip_penalty = message['slip_penalty']
            total_reward = message['total']

            print(f"Progress: {progress_reward:.2f}")
            print(f"On Track: {on_track_reward}")
            print(f"Damage: {car_damage_penalty:.2f}")
            print(f"Orient: {orientation_reward:.2f}")
            print(f"Slip: {slip_penalty:.2f}")
            print(f"Total: {total_reward:.2f}")
            
            sleep(0.5)
            os.system("cls")

            if keyboard.is_pressed("q"):
                print("Stopping.")
                break

if __name__ == "__main__":
    os.system('cls')
    print("Listening for input...")
    receive_data('localhost', 12345)