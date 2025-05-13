import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from kivy.app import App
from kivy.uix.image import Image as KivyImage
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import numpy as np
from scipy.linalg import eigvals
from scipy.stats import entropy
import os
import uuid

def generate_art(file_name):
    # Create a blank image
    img = Image.new('RGB', (800, 800), color='black')
    draw = ImageDraw.Draw(img)

    # Define your color patterns
    colors = [(0, 255, 0), (255, 0, 0), (255, 20, 147)]  # Green, Red, Hot Pink

    # Create a matrix and calculate eigenvalues
    matrix = np.random.rand(3, 3)
    eig_values = eigvals(matrix)

    # Normalize eigenvalues to use as color intensities
    norm_eig_values = np.abs(eig_values) / np.max(np.abs(eig_values))

    # Draw colored rectangles based on eigenvalues
    for i in range(0, 800, 20):
        for j in range(0, 800, 20):
            color = tuple(int(c * norm_eig_values[np.random.randint(0, len(norm_eig_values))]) for c in colors[np.random.randint(0, len(colors))])
            draw.rectangle([i, j, i+20, j+20], fill=color)

    # Draw black lines crossing each other
    for _ in range(50):
        x1, y1 = np.random.randint(0, 800, 2)
        x2, y2 = np.random.randint(0, 800, 2)
        draw.line([x1, y1, x2, y2], fill='black', width=2)

    # Calculate entropy of the image
    img_array = np.array(img)
    img_entropy = entropy(img_array.flatten())
    print(f'Image Entropy: {img_entropy}')

    # Save the image with the user-defined file name
    img.save(f'{file_name}.png')

    # Generate a unique HTTPS link (UUID for simplicity)
    unique_link = f'https://example.com/art/{uuid.uuid4()}'
    print(f'Unique Link: {unique_link}')

    return unique_link

class ArtApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.image = KivyImage(source='art.png')
        layout.add_widget(self.image)

        self.text_input = TextInput(hint_text='Enter file name', multiline=False)
        layout.add_widget(self.text_input)

        btn = Button(text='Generate Art')
        btn.bind(on_press=self.update_image)
        layout.add_widget(btn)

        return layout

    def update_image(self, instance):
        file_name = self.text_input.text
        if file_name:
            unique_link = generate_art(file_name)
            self.image.source = f'{file_name}.png'
            self.image.reload()
            print(f'Art saved as {file_name}.png with link: {unique_link}')
        else:
            print('Please enter a file name.')

if __name__ == '__main__':
    generate_art('art')
    ArtApp().run()
    
    import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

# Define Pauli matrices (keeping this for reference)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Input validation
def validate_input(text):
    if not isinstance(text, str) or not text.isascii():
        raise ValueError("Invalid input: text must be an ASCII string.")

# Encryption using AES
def encrypt_text(plain_text):
    key = get_random_bytes(16)  # AES key must be either 16, 24, or 32 bytes long
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(plain_text.encode('ascii'))
    return base64.b64encode(nonce + ciphertext).decode('ascii'), key

# Decryption using AES
def decrypt_text(encrypted_text, key):
    data = base64.b64decode(encrypted_text)
    nonce = data[:16]
    ciphertext = data[16:]
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plain_text = cipher.decrypt(ciphertext).decode('ascii')
    return plain_text

# User input for the data to encrypt
user_input = input("Enter text to encrypt: ")

# Validate input
validate_input(user_input)

# Encrypt the input text
encrypted_text, encryption_key = encrypt_text(user_input)

# Decrypt the text to verify
decrypted_text = decrypt_text(encrypted_text, encryption_key)

print("Original text:", user_input)
print("Encrypted text:", encrypted_text)
print("Decrypted text:", decrypted_text)
