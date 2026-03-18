import pandas as pd
import csv


def calculate_averages_from_csv(file_path):
    """
    计算CSV文件中数值列的平均值

    参数:
    file_path: CSV文件路径

    返回:
    dict: 包含各列平均值的字典
    """
    try:
        # 读取CSV文件，假设列名在第一行
        # 使用pandas读取，自动处理数值列
        df = pd.read_csv(file_path)

        # 获取数值列（排除非数值列）
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_columns) == 0:
            print("文件中没有找到数值列")
            return None

        # 计算每列的平均值
        averages = {}
        for col in numeric_columns:
            averages[col] = df[col].mean()

        # 打印结果
        print(f"文件: {file_path}")
        print("各列平均值:")
        for col, avg in averages.items():
            print(f"  {col}: {avg:.6f}")

        return averages

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return None
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None


def calculate_averages_manual(file_path):
    """
    手动计算CSV文件中数值列的平均值（不使用pandas）

    参数:
    file_path: CSV文件路径

    返回:
    dict: 包含各列平均值的字典
    """
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)

            # 读取标题行
            headers = next(reader)

            # 初始化列数据和计数器
            column_sums = [0.0] * len(headers)
            column_counts = [0] * len(headers)

            # 处理每一行数据
            row_count = 0
            for row in reader:
                row_count += 1
                for i, value in enumerate(row):
                    try:
                        # 尝试转换为浮点数
                        num = float(value)
                        column_sums[i] += num
                        column_counts[i] += 1
                    except ValueError:
                        # 如果不是数值，跳过
                        continue

            # 计算平均值
            averages = {}
            for i, header in enumerate(headers):
                if column_counts[i] > 0:
                    averages[header] = column_sums[i] / column_counts[i]
                else:
                    averages[header] = None

            # 打印结果
            print(f"文件: {file_path}")
            print(f"总行数: {row_count}")
            print("各列平均值:")
            for header, avg in averages.items():
                if avg is not None:
                    print(f"  {header}: {avg:.6f}")
                else:
                    print(f"  {header}: 无有效数值数据")

            return averages

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return None
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None


def calculate_from_content(content):
    """
    从给定的文本内容直接计算平均值

    参数:
    content: CSV格式的文本内容

    返回:
    dict: 包含各列平均值的字典
    """
    try:
        import io
        # 将内容转换为文件对象
        content_io = io.StringIO(content)
        reader = csv.reader(content_io)

        # 读取标题行
        headers = next(reader)

        # 初始化列数据和计数器
        column_sums = [0.0] * len(headers)
        column_counts = [0] * len(headers)

        # 处理每一行数据
        row_count = 0
        for row in reader:
            row_count += 1
            for i, value in enumerate(row):
                try:
                    # 尝试转换为浮点数
                    num = float(value)
                    column_sums[i] += num
                    column_counts[i] += 1
                except ValueError:
                    # 如果不是数值，跳过
                    continue

        # 计算平均值
        averages = {}
        for i, header in enumerate(headers):
            if column_counts[i] > 0:
                averages[header] = column_sums[i] / column_counts[i]
            else:
                averages[header] = None

        # 打印结果
        print(f"从文本内容计算")
        print(f"总行数: {row_count}")
        print("各列平均值:")
        for header, avg in averages.items():
            if avg is not None:
                print(f"  {header}: {avg:.6f}")
            else:
                print(f"  {header}: 无有效数值数据")

        return averages

    except Exception as e:
        print(f"处理内容时出错: {str(e)}")
        return None


if __name__ == "__main__":
    file_path = "outputs/Covid/2000-fgsm/unet/eval_fgsm_eps0.0157_defense_none.csv"
    calculate_averages_from_csv(file_path)

    # 方法2: 手动从文件计算（不使用pandas）
    # calculate_averages_manual(file_path)
