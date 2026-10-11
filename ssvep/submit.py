# -*- coding: utf-8 -*-
"""标准结果文件生成：把预测类别写入赛题提供的 Excel 模板。

比赛要求：结果记录到赛题提供的标准结果文件（Excel），文件名不可更改。
推荐流程：
    1) 把赛题模板（如 result.xlsx）放到工程目录；
    2) 用本模块把预测结果写入模板对应列，另存/覆盖为原文件名提交。
"""
import numpy as np
import pandas as pd


def write_submission(predictions, out_path, template_path=None,
                     id_column="片段编号", pred_column="识别结果",
                     n_items=None):
    """生成结果 Excel。

    参数：
        predictions   预测类别数组（取值 0~7），长度 = 片段数
        out_path      输出 xlsx 路径（保持赛题要求的文件名）
        template_path 赛题标准结果文件路径：
                      - 给定时，读取模板并把 pred_column 列填为预测结果；
                      - 不给定时，新建两列（id_column, pred_column）。
        id_column/pred_column  模板中对应的列名（拿到模板后按实际表头修改）
        n_items       模板方式下的片段总数（用于校验长度）
    """
    preds = np.asarray(predictions, dtype=int).ravel()

    if template_path is not None:
        df = pd.read_excel(template_path)
        if n_items is not None:
            assert len(preds) == n_items, \
                f"预测数 {len(preds)} 与片段总数 {n_items} 不一致"
        assert len(df) == len(preds), \
            f"模板行数 {len(df)} 与预测数 {len(preds)} 不一致"
        if pred_column not in df.columns:
            # 模板没有该列名时，默认写入最后一列之后
            pred_column = df.columns[-1]
        df[pred_column] = preds
        df.to_excel(out_path, index=False)
    else:
        ids = np.arange(1, len(preds) + 1)
        df = pd.DataFrame({id_column: ids, pred_column: preds})
        df.to_excel(out_path, index=False)

    print(f"结果已写入 {out_path}（共 {len(preds)} 条，类别分布："
          f"{dict(zip(*np.unique(preds, return_counts=True)))}）")
    return df
