import pygame
import random
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os

# Инициализация Pygame
pygame.init()

# Настройки экрана
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Загрузка изображения динозавра
dino_img = pygame.image.load("dino.png")
dino_img = pygame.transform.scale(dino_img, (50, 50))  # Масштабирование

# Динозавр
dino_rect = pygame.Rect(100, HEIGHT - 60, 50, 50)

# Гравитация и скорость
dino_gravity = 0
obstacle_speed = 9
score = 0
double_jump = False

# MediaPipe для детекции руки
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Модель для ИИ
model = Sequential([
    Flatten(input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Данные для обучения ИИ
data_file = "game_data.npy"
if os.path.exists(data_file):
    training_data = np.load(data_file, allow_pickle=True)
else:
    training_data = np.array([])

# Режим игры
mode = 1  # 1 - Игрок, 2 - ИИ, 3 - Управление рукой


def detect_hands():
    """Определение количества рук в кадре"""
    ret, frame = cap.read()
    if not ret:
        return 0
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    cv2.imshow("Hand Tracking", frame)
    if result.multi_hand_landmarks:
        return len(result.multi_hand_landmarks)  # Количество рук
    return 0


def generate_obstacle():
    """Генерация препятствий"""
    width = random.randint(20, 70)
    height = random.randint(30, 120)
    y_position = HEIGHT - height if random.random() > 0.5 else HEIGHT - height - 120  # Наземные и воздушные препятствия
    return pygame.Rect(WIDTH, y_position, width, height)


obstacle_rect = generate_obstacle()
game_data = []

running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                mode = 1
            elif event.key == pygame.K_2:
                mode = 2
            elif event.key == pygame.K_3:
                mode = 3

    if mode == 1:  # Игрок
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and dino_rect.bottom >= HEIGHT - 100:
            dino_gravity = -15
            double_jump = True  # Включаем возможность дабл-прыжка

    elif mode == 2:  # ИИ
        input_data = np.array([[dino_rect.bottom, obstacle_rect.left]])
        action = model.predict(input_data)
        if action > 0.5 and dino_rect.bottom >= HEIGHT - 100:
            dino_gravity = -15
            double_jump = True

    elif mode == 3:  # Управление через камеру
        hands_detected = detect_hands()
        if hands_detected == 1 and dino_rect.bottom >= HEIGHT - 100:
            dino_gravity = -15
            double_jump = True
        elif hands_detected == 2 and double_jump:  # Дабл-прыжок при двух руках
            dino_gravity = -12
            double_jump = False

    # Сохранение данных для обучения ИИ
    game_data.append([dino_rect.bottom, obstacle_rect.left, int(dino_gravity < 0)])

    # Гравитация и прыжки
    dino_gravity += 1
    dino_rect.y += dino_gravity
    if dino_rect.bottom >= HEIGHT - 60:
        dino_rect.bottom = HEIGHT - 60
        double_jump = False  # После приземления сбрасываем возможность дабл-прыжка

    # Движение препятствия
    obstacle_rect.x -= obstacle_speed
    if obstacle_rect.right < 0:
        obstacle_rect = generate_obstacle()
        score += 1
        obstacle_speed += 0.5

    # Проверка столкновения
    if dino_rect.colliderect(obstacle_rect):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Game Over! Score: {score} (R - Restart, Q - Quit)", True, BLACK)
        screen.blit(text, (WIDTH // 4, HEIGHT // 2))
        pygame.display.flip()
        pygame.time.delay(2000)

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        dino_rect = pygame.Rect(100, HEIGHT - 60, 50, 50)
                        obstacle_rect = generate_obstacle()
                        obstacle_speed = 9
                        score = 0
                        waiting = False
                    elif event.key == pygame.K_q:
                        running = False
                        waiting = False

    # Отображение счета
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

    # Отображение элементов
    screen.blit(dino_img, dino_rect)
    pygame.draw.rect(screen, RED, obstacle_rect)

    pygame.display.flip()
    clock.tick(30)

# Сохранение данных игры
np.save(data_file, np.array(game_data, dtype=object))

cap.release()
cv2.destroyAllWindows()
pygame.quit()
