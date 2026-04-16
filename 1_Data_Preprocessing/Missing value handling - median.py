import pandas as pd
import numpy as np

# 步骤1: 导入Excel文件
file_path = 'C:/Users/张梦雨/Desktop/训练集_最终32特征_完美排版版.xlsx'
df = pd.read_excel(file_path)

# 步骤2: 查看缺失值数量
print("原始缺失值数量：")
print(df.isnull().sum())

# === 关键配置 ===
# 假设第1列（索引0）是标签列（例如：0代表对照，1代表病例）
label_col_name = df.columns[0]

# 确定特征列的范围，如再第8列，就是8-2，填6
features_columns = df.columns[1:]

print(f"当前使用的标签列为: {label_col_name}")
print(f"正在处理 {len(features_columns)} 个特征列的缺失值...")

# 步骤3: 按标签组别（Case/Control）分别计算中位数并填充
for col in features_columns:
    # 判断该列是否为数值类型
    if pd.api.types.is_numeric_dtype(df[col]):
        if df[col].isnull().any():  # 只有当该列存在缺失值时才处理

            filled_col = df[col].fillna(df.groupby(label_col_name)[col].transform('median'))

            # 如果分组填充后还有空值（例如某一组全都是NaN），则兜底使用全局中位数
            if filled_col.isnull().any():
                global_median = df[col].median()
                filled_col = filled_col.fillna(global_median)

            df[col] = filled_col
    else:
        # 如果是非数值列（如备注等），跳过
        pass

# 验证填充结果
print("\n填充后缺失值数量：")
print(df.isnull().sum())

# 步骤4: 保存处理后的数据
output_file_path = 'C:/Users/张梦雨/Desktop/训练集.xlsx'
df.to_excel(output_file_path, index=False)

print(f"处理完成！数据已保存至 {output_file_path}")