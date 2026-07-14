# jsonUtils.py     json相关函数

import re

class JsonUtils :
    @staticmethod
    def extract_pattern(text, pattern):
        pattern = re.compile(f"```{pattern}\s(.*?)```", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else text

# if __name__ == "__main__":
#     output = JsonUtils.extract_pattern(text, "markdown")
#     print(output)