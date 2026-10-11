# output 目录

存放预测结果与中间产物：

- `demo_result.xlsx`：`examples/demo_synthetic.py` 自测生成的结果文件示例
- 正式比赛：把赛题提供的**标准结果 Excel 模板**放到工程目录，
  用 `ssvep/submit.py` 的 `write_submission()` 填入预测结果；
  **输出文件名必须与赛题模板一致，不得改名**。
