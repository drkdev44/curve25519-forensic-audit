import os
from nacl.signing import SigningKey

def generate_keys(output_filename, count=500):
    print(f"[+] Generating {count} Ed25519 keys and saving to: {output_filename}")
    with open(output_filename, "wb") as f:
        for _ in range(count):
            sk = SigningKey.generate()
            vk_bytes = bytes(sk.verify_key)
            f.write(vk_bytes)
    print("[✔] Completed!")

if __name__ == "__main__":
    generate_keys("linux_x86.bin", count=500) #i wrote windows_arm for the arm machine!
