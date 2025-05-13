import pygame
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pybullet as p
import pybullet_data
import time
import random
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import threading

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# SQLite database setup
DB_PATH = "/sdcard/robot_memory.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT,
                observation TEXT,
                action TEXT,
                reward REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logging.info(f"Initialized SQLite database at {DB_PATH}")
    except Exception as e:
        logging.error(f"Failed to initialize SQLite database: {str(e)}")
        raise

# Simulated robot environment (PyBullet)
class RobotEnv:
    def __init__(self):
        self.dummy_mode = False
        try:
            p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
            data_path = pybullet_data.getDataPath()
            logging.debug(f"PyBullet data path: {data_path}")
            if not os.path.exists(data_path):
                logging.error(f"PyBullet data path does not exist: {data_path}")
                raise FileNotFoundError("PyBullet data path not found")
            p.setAdditionalSearchPath(data_path)
            urdf_file = "/sdcard/pybullet_data/humanoid.urdf"  # Try humanoid
            logging.debug(f"Checking URDF file: {urdf_file}")
            if not os.path.exists(urdf_file):
                logging.error(f"URDF file not found: {urdf_file}")
                logging.warning("Switching to dummy mode (no simulation)")
                self.dummy_mode = True
                self.robot = None
                self.joint_count = 14
            else:
                self.robot = p.loadURDF(urdf_file, [0, 0, 1])
                self.joint_count = 14
                # Add a heavy object (20kg box)
                self.box = p.loadURDF("/sdcard/pybullet_data/cube.urdf", [1, 0, 0.5], globalScaling=0.5)
                p.changeDynamics(self.box, -1, mass=20.0)
            self.step_count = 0
            self.max_steps = 1000
        except Exception as e:
            logging.error(f"Failed to initialize RobotEnv: {str(e)}")
            logging.warning("Switching to dummy mode (no simulation)")
            self.dummy_mode = True
            self.robot = None
            self.joint_count = 14

    def reset(self):
        if self.dummy_mode:
            self.step_count = 0
            return self._get_obs()
        p.resetSimulation()
        p.setGravity(0, 0, -9.81 * np.random.uniform(0.9, 1.1))
        self.robot = p.loadURDF("/sdcard/pybullet_data/humanoid.urdf", [0, 0, 1])
        self.box = p.loadURDF("/sdcard/pybullet_data/cube.urdf", [1, 0, 0.5], globalScaling=0.5)
        p.changeDynamics(self.box, -1, mass=20.0)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        if self.dummy_mode:
            obs = self._get_obs()
            reward = 1.0
            done = self.step_count >= self.max_steps
            self.step_count += 1
            return obs, reward, done, action
        action = np.clip(action, -1, 1)
        for i in range(self.joint_count):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i], maxVelocity=2.0)
        p.stepSimulation()
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self.step_count >= self.max_steps or self._is_unstable(obs)
        self.step_count += 1
        return obs, reward, done, action

    def _get_obs(self):
        if self.dummy_mode:
            return np.random.randn(20)
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        vel, _ = p.getBaseVelocity(self.robot)
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(self.joint_count)]
        return np.array([pos[2], vel[0]] + joint_states[:18])

    def _compute_reward(self, obs, action):
        if self.dummy_mode:
            return 1.0
        balance = np.abs(obs[0])
        smoothness = np.sum(np.abs(np.diff(action)))
        forward_velocity = max(0, obs[1]) if len(obs) > 1 else 0
        box_height = p.getBasePositionAndOrientation(self.box)[0][2] if hasattr(self, 'box') else 0
        return 1.0 - 0.1 * balance - 0.05 * smoothness + 0.2 * forward_velocity + 0.3 * box_height

    def _is_unstable(self, obs):
        collision = len(p.getContactPoints(self.robot)) > 0 if not self.dummy_mode else False
        return np.abs(obs[0]) > 1.0 or collision

# AI with Random Forest and SQLite memory
class RobotAI:
    def __init__(self):
        try:
            self.env = RobotEnv()
            self.dummy_mode = self.env.dummy_mode
            self.scaler = StandardScaler()
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self.commands = ["Walk Forward", "Walk Backward", "Move Arms", "Move Legs"]
            self.is_training = True
            init_db()
        except Exception as e:
            logging.error(f"Failed to initialize RobotAI: {str(e)}")
            raise

    def save_to_db(self, command, obs, action, reward):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO training_data (command, observation, action, reward)
                VALUES (?, ?, ?, ?)
                ''',
                (command, json.dumps(obs.tolist()), json.dumps(action.tolist()), reward)
            )
            conn.commit()
            conn.close()
            logging.debug(f"Saved training data for command: {command}")
        except Exception as e:
            logging.error(f"Failed to save to SQLite: {str(e)}")

    def load_from_db(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT observation, reward FROM training_data')
            data = cursor.fetchall()
            conn.close()
            if not data:
                return [], []
            X = [json.loads(row[0]) for row in data]
            y = [np.argmax([row[1], -row[1]]) for row in data]
            logging.info(f"Loaded {len(X)} training samples from SQLite")
            return X, y
        except Exception as e:
            logging.error(f"Failed to load from SQLite: {str(e)}")
            return [], []

    def train_skill(self, command):
        if not self.is_training:
            return None
        if self.dummy_mode:
            logging.info(f"Dummy mode: Simulating training for {command}")
            return np.random.uniform(-1, 1, self.env.joint_count) * 0.3
        obs = self.env.reset()
        X, y = [], []
        last_action = None
        for _ in range(500):
            action = np.random.uniform(-1, 1, self.env.joint_count) * 0.3
            if command == "Walk Forward":
                action[6:10] += 0.5
            elif command == "Walk Backward":
                action[6:10] -= 0.5
            elif command == "Move Arms":
                action[0:6] += np.random.uniform(-0.5, 0.5, 6)
            elif command == "Move Legs":
                action[6:10] += np.random.uniform(-0.3, 0.3, 4)
            next_obs, reward, done, action = self.env.step(action)
            self.save_to_db(command, next_obs, action, reward)
            X.append(next_obs)
            y.append(np.argmax([reward, -reward]))
            obs = next_obs
            last_action = action
            if done:
                break
        X_db, y_db = self.load_from_db()
        X.extend(X_db)
        y.extend(y_db)
        if X:
            X = self.scaler.fit_transform(X)
            self.model.fit(X, y)
        logging.info(f"Trained skill: {command} with {len(X)} samples")
        return last_action

    def predict_action(self, obs):
        if self.dummy_mode:
            return np.random.uniform(-1, 1, self.env.joint_count) * 0.3
        X_db, y_db = self.load_from_db()
        if not X_db:
            return np.random.uniform(-1, 1, self.env.joint_count) * 0.3
        X = self.scaler.fit_transform(X_db)
        self.model.fit(X, y_db)
        obs_scaled = self.scaler.transform([obs])
        pred = self.model.predict(obs_scaled)
        action = np.random.uniform(-1, 1, self.env.joint_count) * 0.3
        if pred[0] == 0:
            action += 0.1
        return action

# Reasoning module
def reason_action(command):
    reasons = {
        "Walk Forward": "I'm walking forward to practice balance and coordination.",
        "Walk Backward": "I'm walking backward to improve my reverse movement skills.",
        "Move Arms": "I'm moving my arms to learn joint control and dexterity.",
        "Move Legs": "I'm moving my legs to strengthen my walking foundation."
    }
    return reasons.get(command, "I'm exploring my capabilities.")

# HTTP server
class RobotServer(BaseHTTPRequestHandler):
    ai = None

    def do_POST(self):
        try:
            if RobotServer.ai is None:
                RobotServer.ai = RobotAI()
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            command = json.loads(post_data.decode('utf-8'))['command']
            reason = reason_action(command)
            last_action = RobotServer.ai.train_skill(command)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success', 'reason': reason, 'action': last_action.tolist() if last_action is not None else []}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            logging.error(f"Server error processing request: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'error', 'reason': str(e)}
            self.wfile.write(json.dumps(response).encode('utf-8'))

# Pygame app
class RobotControlApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Robot Control")
        self.font = pygame.font.SysFont("arial", 24)
        self.server_url = "http://localhost:8000"
        self.commands = ["Walk Forward", "Walk Backward", "Move Arms", "Move Legs"]
        self.random_commands = random.choices(self.commands, k=8)
        self.current_command_idx = 0
        self.status = "Ready"
        self.last_action = None
        self.command_start_time = time.time()

    def draw_avatar(self, action):
        self.screen.fill((255, 255, 255))
        center_x, center_y = 400, 300

        if action is not None:
            left_arm_angle = np.mean(action[0:3]) * 90
            right_arm_angle = np.mean(action[3:6]) * 90
            left_leg_angle = np.mean(action[6:8]) * 90
            right_leg_angle = np.mean(action[8:10]) * 90
        else:
            left_arm_angle = right_arm_angle = left_leg_angle = right_leg_angle = 0

        pygame.draw.circle(self.screen, (0, 0, 255), (center_x, center_y - 100), 30)
        pygame.draw.line(self.screen, (0, 0, 0), (center_x, center_y - 70), (center_x, center_y), 5)

        arm_length = 50
        left_arm_end = (
            center_x - arm_length * np.cos(np.radians(left_arm_angle)),
            center_y - 70 + arm_length * np.sin(np.radians(left_arm_angle))
        )
        right_arm_end = (
            center_x + arm_length * np.cos(np.radians(right_arm_angle)),
            center_y - 70 + arm_length * np.sin(np.radians(right_arm_angle))
        )
        pygame.draw.line(self.screen, (255, 0, 0), (center_x - 20, center_y - 70), left_arm_end, 5)
        pygame.draw.line(self.screen, (255, 0, 0), (center_x + 20, center_y - 70), right_arm_end, 5)

        leg_length = 60
        left_leg_end = (
            center_x - leg_length * np.cos(np.radians(left_leg_angle)),
            center_y + leg_length * np.sin(np.radians(left_leg_angle))
        )
        right_leg_end = (
            center_x + leg_length * np.cos(np.radians(right_leg_angle)),
            center_y + leg_length * np.sin(np.radians(right_leg_angle))
        )
        pygame.draw.line(self.screen, (0, 255, 0), (center_x - 10, center_y), left_leg_end, 5)
        pygame.draw.line(self.screen, (0, 255, 0), (center_x + 10, center_y), right_leg_end, 5)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running and self.current_command_idx < len(self.random_commands):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if time.time() - self.command_start_time >= 3:
                if self.current_command_idx < len(self.random_commands):
                    command = self.random_commands[self.current_command_idx]
                    self.send_command(command)
                    self.current_command_idx += 1
                    self.command_start_time = time.time()
                else:
                    self.status = "All commands completed"

            self.draw_avatar(self.last_action)
            status_text = self.font.render(f"Status: {self.status}", True, (0, 0, 0))
            command_text = self.font.render(
                f"Command: {self.random_commands[self.current_command_idx-1] if self.current_command_idx > 0 else 'None'}",
                True, (0, 0, 0)
            )
            self.screen.blit(status_text, (10, 10))
            self.screen.blit(command_text, (10, 40))
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def send_command(self, command):
        logging.info(f"Sending command: {command}")
        try:
            response = requests.post(self.server_url, json={"command": command}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.status = data.get("reason", "Command received")
                self.last_action = np.array(data.get("action", [])) if data.get("action") else None
                logging.debug(f"Last action: {self.last_action}")
            else:
                self.status = f"Server error: {response.status_code}"
                logging.error(f"Server returned status code: {response.status_code}")
        except Exception as e:
            self.status = f"Error: {str(e)}"
            logging.error(f"Request error: {str(e)}")

if __name__ == '__main__':
    try:
        server = HTTPServer(('0.0.0.0', 8000), RobotServer)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        logging.info("Server running on port 8000")

        app = RobotControlApp()
        app.run()
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}")
