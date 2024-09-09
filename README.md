def relu(x):
    return np.maximum(0, x)

def calculate_layer_output(inputs, weights, bias):
    # 가중합 계산
    weighted_sum = np.dot(inputs, weights) + bias
    # 활성화 함수 적용
    output = relu(weighted_sum)
    return output

def print_layer_outputs(inputs, weights1, bias1, weights2, bias2, weights3, bias3):
    # 첫 번째 레이어의 출력 계산
    layer1_output = calculate_layer_output(inputs, weights1, bias1)
    print("First Layer Output:")
    print(layer1_output)
    
    # 두 번째 레이어의 출력 계산
    layer2_output = calculate_layer_output(layer1_output, weights2, bias2)
    print("\nSecond Layer Output:")
    print(layer2_output)
    
    # 세 번째 레이어의 출력 계산
    layer3_output = calculate_layer_output(layer2_output, weights3, bias3)
    print("\nThird Layer Output:")
    print(layer3_output)

# 입력 데이터와 가중치 설정
inputs = np.array([
    [1, 2],   # 샘플 1
    [3, -1],  # 샘플 2
    [2, 4],   # 샘플 3
    [0, 1],   # 샘플 4
    [5, -3]   # 샘플 5
])

# 첫 번째 레이어의 가중치와 편향
weights1 = np.array([0.8, -0.5])
bias1 = 0.5

# 두 번째 레이어의 가중치와 편향
# 첫 번째 레이어의 출력 차원과 일치
weights2 = np.array([1.2, -0.8])
bias2 = -0.2

# 세 번째 레이어의 가중치와 편향
# 두 번째 레이어의 출력 차원과 일치
weights3 = np.array([0.6, 0.9])
bias3 = 0.1

# 결과 출력
print_layer_outputs(inputs, weights1, bias1, weights2, bias2, weights3, bias3)
