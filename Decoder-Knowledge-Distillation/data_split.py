import json
import random

def main():
    data_path = "./data/alpaca_data_cleaned.json"
    test_ratio = 0.05

    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 打乱
    random.shuffle(data)

    # 划分
    test_size = int(len(data) * test_ratio)
    test_data = data[:test_size]
    train_data = data[test_size:]

    # 保存
    with open("./data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open("./data/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"总数: {len(data)}, 训练集: {len(train_data)}, 测试集: {len(test_data)}")

if __name__ == "__main__":
    main()
