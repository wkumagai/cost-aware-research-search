# Architecture: Cost-Aware Research Search

## 現在実装されているアルゴリズム

### Feedback-Driven Sequential Improvement (v0.1)

現在のメインループ (`src/loop.py`) は以下のアルゴリズムで動作する。

```
┌───────────────────────────────────────────────────────┐
│  Iteration 1                                          │
│  ┌──────────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ Claude Sonnet ├──►│ Stage 0  ├──►│ Code Gen     │  │
│  │ Idea Spec Gen│   │ Feasible?│   │ (Claude)     │  │
│  └──────────────┘   └──────────┘   └──────┬───────┘  │
│                                           ▼           │
│  ┌──────────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ GPT-5.4-pro  │◄──┤ Results  │◄──┤ Execute      │  │
│  │ Judge        │   │ JSON     │   │ (subprocess) │  │
│  └──────┬───────┘   └──────────┘   └──────────────┘  │
│         │ feedback                                    │
│         ▼                                             │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Improvement Suggestions + Key Finding            │ │
│  └──────────────────────────┬───────────────────────┘ │
└─────────────────────────────┼─────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────┐
│  Iteration 2                                          │
│  ┌──────────────┐                                     │
│  │ Claude Sonnet ├── 前回のspec + feedback を入力     │
│  │ Improved Spec│   → 改善されたidea specを生成      │
│  └──────────────┘                                     │
│        ... (同じパイプライン)                          │
└───────────────────────────────────────────────────────┘
```

### 各ステージの役割

| Stage | 名前 | 何をするか | 失敗時 |
|-------|------|-----------|--------|
| 0 | Static Feasibility | YAML検証、仮説・指標の存在確認、CPU制約チェック | spec再生成 |
| 1 | Code Generation | Claude がアイデアを実行可能なPythonに変換 | — |
| 2 | Execution | subprocess で実験コードを実行、結果JSONを取得 | コード修復(repair)を試行 |
| — | Judging | GPT-5.4-pro がスコア + 改善提案を返す | フォールバックスコア |
| — | Improvement | 前回のspec + feedback → Claude が改善版specを生成 | — |

### 修復 (Repair)

実験コードが失敗した場合：
1. stderrの最終10行をClaude に渡す
2. 「このエラーを修正して」と再生成を依頼
3. 修復コードを再実行
4. それでも失敗 → ダミースコア(2/10) + 「簡略化せよ」のフィードバックで次イテレーションへ

### コスト構造

| コンポーネント | モデル | 1回あたりのコスト |
|--------------|--------|-----------------|
| Idea Spec生成 | Claude Sonnet 4 | ~$0.01 |
| Code生成 | Claude Sonnet 4 | ~$0.02 |
| 実験実行 | 各種 (gpt-4o-mini等) | ~$0.01-0.10 |
| **Judge** | **GPT-5.4-pro** | **~$0.10** |
| Repair (発生時) | Claude Sonnet 4 | ~$0.02 |

**4イテレーションの総コスト: ~$1-2**
（ジャッジがコストの60-70%を占める）

---

## 今後のアルゴリズム拡張

### Phase 2: Thompson Sampling による方向選択

複数の研究方向（arm）を持ち、Beta分布でexplore/exploit。

```
Arms: [prompt_eng, embedding, model_behavior, ...]
  → Thompson Sampling で方向を選択
  → その方向で idea spec を生成
  → 実験 → judge → reward で Beta を更新
```

### Phase 3: MCTS による階層的探索

アイデアをツリーとして展開し、UCBで有望なノードを選択。

### Phase 4: Offline RL

蓄積されたログ（idea → experiment → score）から
mutation policy を学習。
