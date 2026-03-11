# Cost-Aware Research Search

構造化された研究アイデア探索と、段階的な実験実行を行うためのリポジトリです。

このプロジェクトは、研究アイデアを単なる自由文の発想としてではなく、実行可能性・計算コスト・時間制約・失敗回復を含む探索問題として扱います。
目的は、実装可能で、実験で一定の改善が見込めて、最終的に論文化候補になりうるアイデアを効率よく見つけることです。

---

## 目的

このリポジトリの目的は次の3つです。

1. 研究アイデアを構造化された候補として表現する
2. 候補を安価な評価から高価な評価へ段階的に選別する
3. 失敗した候補も記録し、修復や次回探索に再利用する

---

## 基本方針

- 自由文のアイデアをそのまま実行しない
- すべての候補を typed idea spec に変換する
- cheap test を厚くし、full 実験に送る候補を絞る
- 良い候補だけでなく、異なる方向性の候補も保持する
- failure を捨てず、分類・記録・再利用する
- 最初から強化学習を主役にせず、まずは探索と評価基盤を作る

---

## 全体像

```
Problem Definition
-> Idea Spec Generator
-> Search Controller
-> Multi-Fidelity Evaluator
-> Repair Manager
-> Archive
-> Judge
-> Logs / Artifacts
```

---

## 各コンポーネント

### 1. Idea Spec Generator

研究アイデアを、実行可能な構造化フォーマットに変換します。

例:
- 仮説
- どのモジュールを変えるか
- 何を評価するか
- どんな失敗が起きそうか

### 2. Search Controller

どの候補を次に試すかを決めます。
候補の展開、変異、昇格、打ち切りを管理します。

### 3. Multi-Fidelity Evaluator

候補を段階的に評価します。

- Stage 0: static feasibility
- Stage 1: smoke test
- Stage 2: proxy experiment
- Stage 3: full experiment
- Stage 4: robustness check

### 4. Repair Manager

失敗した候補に対して、
- 環境修復
- コード修正
- パラメータ縮退
- 局所変異

などを行います。

### 5. Archive

良い候補を保存します。
また、似た候補ばかりに偏らないよう、多様性も管理します。

### 6. Judge

実験結果が近い候補について、補助的に
- 新規性
- 妥当性
- 論文化可能性

などを評価します。

### 7. Logs / Artifacts

すべての試行を記録し、再現・再利用可能にします。

---

## 候補の考え方

このプロジェクトでは、研究アイデアを自由文ではなく、次のような構造化候補として扱います。

- hypothesis
- intervention family
- target component
- implementation scope
- proxy evaluation
- full evaluation
- expected failure modes
- rollback plan

これにより、
- 実装可能性を上げる
- 自動評価しやすくする
- failure recovery をしやすくする

ことを狙います。

---

## 評価の流れ

候補は次の順で評価されます。

### Stage 0: Static Feasibility
- schema validation
- import test
- dependency check
- lint / type check

### Stage 1: Smoke Test
- 最小実行
- 1 batch
- 1 short epoch

### Stage 2: Proxy Experiment
- 短い学習
- 小さいデータ
- 代理指標で比較

### Stage 3: Full Experiment
- 本番に近い条件で実験
- ベースライン比較
- 必要なら ablation

### Stage 4: Robustness Check
- 追加 seed
- 近傍条件
- 再現性・安定性確認

---

## 失敗の扱い

失敗は重要なデータです。
主に次のように分類して扱います。

- dependency / environment error
- syntax / API error
- data pipeline error
- OOM / timeout
- metric bug
- no learning signal
- proxy-only success

各失敗はログに残し、修復できるものは修復し、修復できないものは次回探索に活かします。

---

## 想定する探索スタック

このリポジトリでは、探索アルゴリズムを単独で使うのではなく、役割ごとに使い分ける前提です。

### 外側の探索
- Progressive Widening MCTS

### 予算配分
- Hyperband
- BOHB
- DEHB

### 条件付き・混合探索
- SMAC3
- CoCaBO
- Bounce

### 多様性維持
- MAP-Elites
- BOP-Elites
- Monte Carlo Elites

### operator 選択
- Thompson Sampling
- CORRAL

### 後段の学習
- offline RL

現時点では、まずは探索・評価・ログ基盤を優先し、RL は十分なログが溜まってから導入します。

---

## リポジトリ構成案

```
.
├─ README.md
├─ docs/
│  ├─ architecture.md
│  ├─ experiment-design.md
│  ├─ search-taxonomy.md
│  └─ failure-taxonomy.md
├─ configs/
│  ├─ search_controller.yaml
│  ├─ fidelity_schedule.yaml
│  ├─ archive_schema.yaml
│  └─ judge_rubric.yaml
├─ templates/
│  ├─ idea_spec.yaml
│  ├─ run_log.md
│  └─ candidate_summary.json
├─ logs/
│  ├─ runs/
│  ├─ archive_snapshots/
│  └─ failure_memory/
├─ src/
│  ├─ idea_generation/
│  ├─ search/
│  ├─ evaluation/
│  ├─ repair/
│  ├─ archive/
│  ├─ judge/
│  └─ logging/
└─ results/
   ├─ finalists/
   └─ papers/
```

---

## 最初に作るべきもの

最初の実装では、すべてを一度に作る必要はありません。
まずは次の順で十分です。

### Phase 1
- typed idea spec の定義
- Stage 0〜2 evaluator
- simple search controller
- run log の保存

### Phase 2
- archive
- failure taxonomy
- repair manager
- candidate promotion / termination

### Phase 3
- BOHB / DEHB
- judge
- operator selection

### Phase 4
- long-term memory
- warm-start
- offline RL

---

## 成功条件

このリポジトリの初期成功条件は、次のように定義します。

1. 研究アイデアを構造化候補として保存できる
2. 候補を Stage 0〜2 まで自動で回せる
3. failure を分類してログできる
4. promising candidate を full 実験候補として選べる
5. 実験履歴を次回探索に再利用できる

---

## 今後の拡張候補

- archive による多様性制御
- failure-aware repair policy
- judge panel の導入
- multi-task transfer
- offline RL による mutation policy 学習
- paper draft 自動生成
- GitHub Actions / experiment orchestration 連携

---

## 位置づけ

このリポジトリは、研究自動化を一発で完成させるものではなく、まずは以下を安定化するための基盤です。

- アイデアの構造化
- 実験候補の選別
- cheap-to-expensive evaluation
- failure recovery
- ログと再利用

その上で、将来的により高度な探索や学習を載せていきます。

---

## 一言でいうと

このプロジェクトは、

**「研究アイデア探索」を
自由文生成ではなく、
コスト制約つき・段階評価つき・失敗回復つきの構造化探索として扱う**

ための基盤です。
