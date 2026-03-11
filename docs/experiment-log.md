# Experiment Log

## 全ラン比較

4回のフロー改善を行い、それぞれ4イテレーションのループを実行した。

| Run | Judge | Algorithm | Scores | Best | 主な改善点 |
|-----|-------|-----------|--------|------|-----------|
| v1 | gpt-4o | Sequential | 6→4→6→3 | 6 | (ベースライン) |
| v2 | GPT-5.4-pro | Sequential | 3→3→2→3 | 3 | GPT-5.4-proの判定は厳格 |
| v3 | GPT-5.4-pro | +Thompson Sampling +多様性制約 | 3→2→**5**→2 | **5** | 4つの異なるテーマを探索 |
| v4 | GPT-5.4-pro | +堅牢化コード生成 +ライブラリ制限 | 4→**5**→2→2 | **5** | Iter2で仮説支持、実行安定化 |

### スコア推移グラフ

```
Score
 6 │  ●           ●
 5 │                          ●              ●
 4 │      ●                       ●              ●
 3 │          ●   ●  ●  ●  ●
 2 │                      ●           ●  ●           ●  ●
 1 │
   └──1──2──3──4──1──2──3──4──1──2──3──4──1──2──3──4──
       v1 (gpt-4o)  v2 (5.4-pro) v3 (+TS)     v4 (+robust)
```

---

## Run v1: ベースライン (gpt-4o judge) - 20260311_182717

| Iter | Score | Direction | Finding |
|------|-------|-----------|---------|
| 1 | 6 | prompt_engineering | JSON serialization エラーで実験未完了 |
| 2 | 4 | prompt_engineering | フォーマットによる一貫性改善は見られず |
| 3 | 6 | prompt_engineering | CoTでaccuracy向上確認 |
| 4 | 3 | prompt_engineering | 制約キーワードの効果は微小 |

**問題**: 同じテーマに固着、gpt-4oの判定が甘い

---

## Run v2: GPT-5.4-pro judge - 20260311_183804

| Iter | Score | Direction | Finding |
|------|-------|-----------|---------|
| 1 | 3 | embedding | CoTのembeddingクラスタリング品質: 微小な差のみ |
| 2 | 3 | embedding | Silhouette 0.081→0.124、閾値未達 |
| 3 | 2 | embedding | JSON serialization エラーで部分実行のみ |
| 4 | 3 | embedding | ドメイン特化: 結果に一貫性なし |

**問題**: テーマ固着（4回ともembedding）、フィードバック未伝搬

---

## Run v3: +Thompson Sampling +多様性制約 - 20260311_200319

| Iter | Score | Direction | Finding |
|------|-------|-----------|---------|
| 1 | 3 | embedding_analysis | antonymペアのembedding幾何学: 指標が崩壊 |
| 2 | 2 | model_comparison | few-shot順序効果: サンプル8のみで不十分 |
| 3 | **5** | **embedding_analysis** | **パラフレーズペアの距離が非関連ペアより大幅に近い (0.361 vs 0.970)** |
| 4 | 2 | prompt_engineering | OpenAI APIバージョン非互換で失敗 |

**改善**: Thompson Samplingで4方向を探索。Iter 3で初の5/10到達
**残存問題**: サンプル不足、APIバージョン互換性

---

## Run v4: +堅牢化コード生成 - 20260311_202635

| Iter | Score | Direction | Finding |
|------|-------|-----------|---------|
| 1 | 4 | embedding_analysis | ドメインラベルはembedding空間で回復不可 |
| 2 | **5** | **text_statistics** | **Wikipedia vs GPT生成テキストで7/8特徴量が有意差 (KS>0.3)** |
| 3 | 2 | reasoning_analysis | 推論エラーパターンにクラスタリングなし |
| 4 | 2 | prompt_engineering | APIコードエラーでデータ収集前に失敗 |

**改善**:
- Iter 2で仮説支持の成功実験 (5/10)
- 全4方向が異なるテーマ
- ライブラリ制限によりimportエラー解消
- JSON serialization堅牢化が効果あり

**残存問題**:
- Iter 4のAPIバージョン互換 (openai古いAPI使用)
- スコアの改善傾向がまだ弱い

---

## 改善の効果まとめ

### 機能した改善

| 改善 | 効果 |
|------|------|
| Thompson Sampling + 多様性制約 | テーマ固着解消: v2の全4回同テーマ → v4の全4回異テーマ |
| ライブラリ制限 (allowlist) | import_error解消 (v3で発生 → v4で未発生) |
| Stage 0 サンプルサイズチェック | n_samples < 50 をブロック (v2: 15サンプル → v4: 50+) |
| コード品質ルール (JSON, try/except) | JSON serialization エラー減少 |
| フィードバックのコード生成注入 | 判定の改善提案がコード生成に反映開始 |

### まだ効いていない改善

| 改善 | 問題 |
|------|------|
| フィードバックループによるスコア上昇 | 方向が変わると前回のフィードバックが活きない |
| 修復メカニズム | 成功率50%程度、根本的なAPIバージョン問題は修復不可 |
| 失敗メモリ | まだ蓄積量が少なく効果が限定的 |

---

## 成功した実験の特徴分析

5/10を獲得した2つの実験に共通する特徴:

1. **実験が最後まで完走した** (結果テーブルが出力された)
2. **事前定義した成功閾値との比較があった**
3. **ベースラインとの数値比較が明確だった**
4. **サンプルサイズが50以上だった**

失敗した実験の共通パターン:
- APIバージョン非互換 (openai v0系のコード生成)
- サンプル収集が途中で止まる
- JSON serialization エラー

---

## ファイル構成

```
logs/runs/
├── 20260311_182717/   # v1 (gpt-4o judge, sequential)
├── 20260311_183804/   # v2 (GPT-5.4-pro judge, sequential)
├── 20260311_200319/   # v3 (+Thompson Sampling, +diversity)
└── 20260311_202635/   # v4 (+robustified code gen)
```

---

## 次の改善計画

### 即効性のある改善
1. **OpenAI APIバージョン指定**: コード生成プロンプトに「openai v1+ APIを使え」を明記
2. **成功パターンテンプレート**: 過去の5/10達成コードをテンプレートとして注入
3. **同一方向での改善優先**: 方向が変わっても前回のベストアイデアを参照可能に

### 構造的改善
4. **Multi-fidelity evaluation**: Stage 1 (smoke test) を追加し、コード品質を事前チェック
5. **Archive**: ベストアイデアを保存し、warm-startに利用
6. **Offline RL**: 蓄積されたログから成功パターンを学習
