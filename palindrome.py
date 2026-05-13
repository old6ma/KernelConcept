"""
字符串反转和回文检测模块
"""


def reverse_string(s: str) -> str:
    """反转字符串"""
    return s[::-1]


def is_palindrome(s: str) -> bool:
    """
    检测字符串是否为回文

    忽略大小写和非字母数字字符
    """
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def find_palindromes_in_list(strings: list[str]) -> list[str]:
    """从字符串列表中找出所有回文"""
    return [s for s in strings if is_palindrome(s)]


if __name__ == "__main__":
    # 测试字符串反转
    test_strings = ["hello", "world", "python", "上海自来水来自海上"]
    print("=== 字符串反转测试 ===")
    for s in test_strings:
        print(f"{s} -> {reverse_string(s)}")

    # 测试回文检测
    print("\n=== 回文检测测试 ===")
    palindrome_tests = [
        "racecar",
        "A man a plan a canal Panama",
        "上海自来水来自海上",
        "hello",
        "Was it a car or a cat I saw"
    ]
    for s in palindrome_tests:
        result = "✓ 是回文" if is_palindrome(s) else "✗ 不是回文"
        print(f"'{s}' -> {result}")

    # 测试列表回文查找
    print("\n=== 列表回文查找测试 ===")
    sample_list = ["racecar", "hello", "level", "world", "madam", "python"]
    palindromes = find_palindromes_in_list(sample_list)
    print(f"输入: {sample_list}")
    print(f"回文: {palindromes}")
