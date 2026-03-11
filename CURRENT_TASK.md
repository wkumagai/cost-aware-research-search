# Current Task

- task_id: improve-loop-v2
- system_type: research_search
- summary: 研究探索ループの改善（フィードバック伝搬、テーマ多様性、サンプルサイズ強制）
- goal: スコアがイテレーションごとに改善するループを実現する
- success_criteria:
  - 4イテレーションでスコアが単調増加または最終スコア >= 5/10
  - テーマが固着しない（3連続同一テーマ禁止）
  - サンプルサイズが常に50以上
  - コード実行成功率 >= 75%
- allowed_paths:
  - src/
  - templates/
  - configs/
  - logs/
  - results/
  - docs/

## Identified Problems (from Run 1 & 2)

1. **Feedback not propagating to code gen**: Judge says "increase samples" but code gen ignores it
2. **Theme fixation**: Same topic repeated 4 times without exploring alternatives
3. **Sample size always too small**: 15-40 samples when judge demands 100+
4. **Stage 0 bug**: `passed` field always False (FIXED)
5. **Weak repair**: Just passing error message isn't enough
6. **No search strategy**: Pure sequential improvement without exploration

## Required Improvements

### P0 (Must have)
- Pass judge feedback directly into code gen prompt (not just idea spec)
- Enforce minimum sample size (50+) in code gen
- Add Thompson Sampling over research directions
- Fix Stage 0 to actually gate progression

### P1 (Should have)
- Failure memory: record past failures, inject into code gen to avoid repeats
- Diversity constraint: no 3x same direction in a row
- Better repair: classify error type, apply targeted fix template

### P2 (Nice to have)
- Archive of best ideas with diversity
- Warm-start from previous run's best spec
