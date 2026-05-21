#!/usr/bin/env python3
# sim_heatmap_kv_switch.py  训练+热图+日志+匹配+Few-shot输出
import json, mmh3, numpy as np, argparse, copy, os
from pathlib import Path

DEFAULT_DB_PREFIX = Path(__file__).resolve().parent / "known_attack_result"


def cosine_similarity(left, right):
    left = np.asarray(left, dtype="float32")
    right = np.asarray(right, dtype="float32")
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    denom = np.maximum(left_norm * right_norm, 1e-12)
    return np.dot(left, right.T) / denom

# ---------- 向量化 ----------
def ioc2vec(ioc: dict, use_kv: bool = False) -> np.ndarray:
    z = [0.0] * 32
    for k, v in sorted(ioc.items()):
        z[mmh3.hash(k) % 32] += 1.0
        if use_kv:
            z[mmh3.hash(str(v)) % 32] += 1.0
    z = np.log1p(z) / 10.0
    return np.array(z, dtype='float32')

# ---------- 找 label ----------
def find_label(item: dict) -> str:
    if "json" in item and "label" in item["json"]:
        return item["json"]["label"]["official"]
    if "label" in item:
        return item["label"]["official"]
    raise KeyError("no label.official")

# ---------- 找 IOC ----------
def find_ioc(item: dict) -> dict:
    ioc = None

    if "json" in item:
        if isinstance(item["json"], dict) and "json" in item["json"]:
            ioc = item["json"]["json"]
        else:
            ioc = item["json"]
    else:
        ioc = item

    if isinstance(ioc, list):
        if len(ioc) == 0:
            raise KeyError("IOC list is empty")
        ioc = ioc[0]

    if not isinstance(ioc, dict):
        raise KeyError(f"IOC is not a dict, got {type(ioc)}")

    return ioc

# ---------- 训练+保存 ----------
def build_and_save(data_path, out_prefix, use_kv):
    import tqdm

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vecs, fams, raw_items = [], [], []
    for item in tqdm.tqdm(data, desc="training"):
        try:
            fam = find_label(item)
            ioc = find_ioc(item)
        except KeyError:
            continue
        vecs.append(ioc2vec(ioc, use_kv))
        fams.append(fam)
        raw_items.append(copy.deepcopy(item))

    Z = np.stack(vecs)
    suffix = "_kv" if use_kv else "_key"
    np.save(f"{out_prefix}{suffix}_Z.npy", Z)
    json.dump(fams, open(f"{out_prefix}{suffix}_fams.json", 'w', encoding='utf-8'))
    json.dump(raw_items, open(f"{out_prefix}{suffix}_raw.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"✅ 向量库已保存：{out_prefix}{suffix}_Z.npy")
    print(f"✅ 标签库已保存：{out_prefix}{suffix}_fams.json")
    print(f"✅ 原始数据已保存：{out_prefix}{suffix}_raw.json")

    sim = cosine_similarity(Z)
    plot_and_log(sim, fams, Z, out_prefix, use_kv)
    return Z, fams, raw_items

# # ---------- 匹配 + Few-shot输出（多样本结果直接拼接） ----------
# def match_single(db_prefix, new_path, top_k, use_kv, fewshot_out=None, train_data_path=None):
    
#     suffix = "_kv" if use_kv else "_key"
#     Z_db = np.load(f"{db_prefix}{suffix}_Z.npy")
#     fams = json.load(open(f"{db_prefix}{suffix}_fams.json", encoding='utf-8'))

#     raw_items = None
#     try:
#         raw_items = json.load(open(f"{db_prefix}{suffix}_raw.json", encoding='utf-8'))
#     except FileNotFoundError:
#         pass

#     if raw_items is None and train_data_path:
#         try:
#             with open(train_data_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             raw_items = []
#             for item in data:
#                 try:
#                     _ = find_label(item)
#                     _ = find_ioc(item)
#                     raw_items.append(copy.deepcopy(item))
#                 except KeyError:
#                     continue
#         except Exception as e:
#             print(f"⚠️ 从训练数据加载失败：{e}")

#     if raw_items is None:
#         print("⚠️ 警告：未找到原始数据，Few-shot输出将不可用。")

#     # 加载未知攻击库（支持单条或列表）
#     with open(new_path, encoding='utf-8') as f:
#         new_data = json.load(f)
    
#     # 统一转为列表处理
#     if not isinstance(new_data, list):
#         new_data = [new_data]
    
#     print(f"\n📦 未知攻击库包含 {len(new_data)} 个样本，开始批量匹配...")
    
#     # 直接拼接所有检索结果
#     all_fewshot_results = []
    
#     for idx, new_item in enumerate(new_data):
#         print(f"\n🔍 样本 [{idx+1}/{len(new_data)}]")
        
#         try:
#             z_new = ioc2vec(find_ioc(new_item), use_kv).reshape(1, -1)
#         except KeyError as e:
#             print(f"  ❌ 跳过：无法提取IOC ({e})")
#             continue
            
#         sims = cosine_similarity(z_new, Z_db)[0]
#         top_idx = np.argsort(sims)[::-1][:top_k]

#         print(f"  Top-{top_k} 最相似样本：")
#         for rank, tidx in enumerate(top_idx, 1):
#             # 从 raw_items 提取信息
#             item_data = raw_items[tidx] if raw_items and tidx < len(raw_items) else {}
            
#             # 提取 meta 字段
#             meta = item_data.get("meta", {}) if isinstance(item_data, dict) else {}
#             category = meta.get("category", "N/A") if isinstance(meta, dict) else "N/A"
#             subcategory = meta.get("subcategory", "N/A") if isinstance(meta, dict) else "N/A"
            
#             official = fams[tidx]
            
#             print(f"    [{rank}] cos={sims[tidx]:.3f}  official={official:20s}  category={category:15s}  subcategory={subcategory}")

#         # 直接追加到总列表
#         if raw_items:
#             fewshot_list = [raw_items[tidx] for tidx in top_idx if tidx < len(raw_items)]
#             all_fewshot_results.extend(fewshot_list)
#             print(f"  ✅ 追加 {len(fewshot_list)} 个样本")

#     # 输出纯列表
#     if fewshot_out and all_fewshot_results:
#         with open(fewshot_out, 'w', encoding='utf-8') as f:
#             json.dump(all_fewshot_results, f, ensure_ascii=False, indent=2)
#         print(f"\n✅ 全部完成！共 {len(new_data)} 个未知样本，检索到 {len(all_fewshot_results)} 条记录，已保存至：{fewshot_out}")
#     elif not fewshot_out:
#         print(f"\n📋 检索结果列表（共 {len(all_fewshot_results)} 条）：")
#         print(json.dumps(all_fewshot_results, ensure_ascii=False, indent=2))
    
#     return all_fewshot_results

# ---------- 匹配 + Few-shot输出（多样本结果去重后拼接） ----------
def match_single(db_prefix, new_path, top_k, use_kv, fewshot_out=None, train_data_path=None):
    suffix = "_kv" if use_kv else "_key"
    Z_db = np.load(f"{db_prefix}{suffix}_Z.npy")
    fams = json.load(open(f"{db_prefix}{suffix}_fams.json", encoding='utf-8'))

    raw_items = None
    try:
        raw_items = json.load(open(f"{db_prefix}{suffix}_raw.json", encoding='utf-8'))
    except FileNotFoundError:
        pass

    if raw_items is None and train_data_path:
        try:
            with open(train_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            raw_items = []
            for item in data:
                try:
                    _ = find_label(item)
                    _ = find_ioc(item)
                    raw_items.append(copy.deepcopy(item))
                except KeyError:
                    continue
        except Exception as e:
            print(f"⚠️ 从训练数据加载失败：{e}")

    if raw_items is None:
        print("⚠️ 警告：未找到原始数据，Few-shot输出将不可用。")

    # 加载未知攻击库（支持单条或列表）
    with open(new_path, encoding='utf-8') as f:
        new_data = json.load(f)
    
    # 统一转为列表处理
    if not isinstance(new_data, list):
        new_data = [new_data]
    
    print(f"\\n📦 未知攻击库包含 {len(new_data)} 个样本，开始批量匹配...")
    
    # ========== 修改1：用set收集所有召回的索引，自动去重 ==========
    unique_indices = set()
    sample_match_info = []  # 记录每个样本匹配了哪些索引，用于日志输出
    
    for idx, new_item in enumerate(new_data):
        print(f"\\n🔍 样本 [{idx+1}/{len(new_data)}]")
        
        try:
            z_new = ioc2vec(find_ioc(new_item), use_kv).reshape(1, -1)
        except KeyError as e:
            print(f"  ❌ 跳过：无法提取IOC ({e})")
            sample_match_info.append([])
            continue
            
        sims = cosine_similarity(z_new, Z_db)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        
        # 记录当前样本的匹配索引
        current_matches = []
        print(f"  Top-{top_k} 最相似样本：")
        for rank, tidx in enumerate(top_idx, 1):
            # 从 raw_items 提取信息
            item_data = raw_items[tidx] if raw_items and tidx < len(raw_items) else {}
            
            # 提取 meta 字段
            meta = item_data.get("meta", {}) if isinstance(item_data, dict) else {}
            category = meta.get("category", "N/A") if isinstance(meta, dict) else "N/A"
            subcategory = meta.get("subcategory", "N/A") if isinstance(meta, dict) else "N/A"
            
            official = fams[tidx]
            
            print(f"    [{rank}] cos={sims[tidx]:.3f}  official={official:20s}  category={category:15s}  subcategory={subcategory}")
            
            # 加入去重集合
            unique_indices.add(int(tidx))
            current_matches.append(int(tidx))
        
        sample_match_info.append(current_matches)
        print(f"  ✅ 本样本召回 {len(current_matches)} 个，当前累计去重后 {len(unique_indices)} 个")

    # ========== 修改2：根据去重后的索引列表生成最终结果 ==========
    sorted_unique_indices = sorted(list(unique_indices))
    
    print(f"\\n{'='*60}")
    print(f"📊 去重统计：")
    print(f"   原始召回总数：{sum(len(m) for m in sample_match_info)} 条")
    print(f"   去重后总数：{len(sorted_unique_indices)} 条")
    print(f"   去重率：{(1 - len(sorted_unique_indices)/max(sum(len(m) for m in sample_match_info), 1))*100:.1f}%")
    
    # 生成去重后的few-shot结果
    if raw_items:
        dedup_fewshot_list = [raw_items[idx] for idx in sorted_unique_indices if idx < len(raw_items)]
        
        # 输出纯列表
        if fewshot_out:
            with open(fewshot_out, 'w', encoding='utf-8') as f:
                json.dump(dedup_fewshot_list, f, ensure_ascii=False, indent=2)
            print(f"\\n✅ 去重后的Few-shot结果（{len(dedup_fewshot_list)}条）已保存至：{fewshot_out}")
        else:
            print(f"\\n📋 去重后的检索结果列表（共 {len(dedup_fewshot_list)} 条）：")
            print(json.dumps(dedup_fewshot_list, ensure_ascii=False, indent=2))
        
        return dedup_fewshot_list
    
    return []


def _load_vector_db(db_prefix=DEFAULT_DB_PREFIX, use_kv=False):
    suffix = "_kv" if use_kv else "_key"
    Z_db = np.load(f"{db_prefix}{suffix}_Z.npy")
    fams = json.load(open(f"{db_prefix}{suffix}_fams.json", encoding="utf-8"))
    raw_items = json.load(open(f"{db_prefix}{suffix}_raw.json", encoding="utf-8"))
    return Z_db, fams, raw_items


def retrieve_similar_items(sample_data, db_prefix=DEFAULT_DB_PREFIX, top_k=3, use_kv=False):
    if top_k <= 0:
        return []

    Z_db, fams, raw_items = _load_vector_db(db_prefix, use_kv)
    z_new = ioc2vec(find_ioc(sample_data), use_kv).reshape(1, -1)
    sims = cosine_similarity(z_new, Z_db)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for tidx in top_idx:
        if int(tidx) >= len(raw_items):
            continue
        results.append({
            "similarity": float(sims[tidx]),
            "official": fams[tidx],
            "item": raw_items[int(tidx)],
        })
    return results


def _format_case(result, index):
    item = result.get("item", {})
    meta = item.get("meta", {})
    label = item.get("label", {})
    reasoning = item.get("reasoning", {})
    response = item.get("response", {})
    payload = {
        "similarity": round(result.get("similarity", 0.0), 4),
        "meta": meta,
        "label": {
            "official": label.get("official"),
            "severity": label.get("severity"),
        },
        "reasoning": {
            "evidence": reasoning.get("evidence", []),
            "analysis": reasoning.get("analysis", ""),
        },
        "response": {
            "action": response.get("action"),
            "reason": response.get("reason", ""),
        },
    }
    return f"Example {index}:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def build_rag_context(sample_data, db_prefix=DEFAULT_DB_PREFIX, top_k=3, use_kv=False):
    """Return a prompt-ready reference block for an unknown-attack sample."""
    try:
        results = retrieve_similar_items(sample_data, db_prefix=db_prefix, top_k=top_k, use_kv=use_kv)
    except (KeyError, FileNotFoundError, ValueError):
        return ""
    if not results:
        return ""
    return "\n\n".join(_format_case(result, idx) for idx, result in enumerate(results, 1))


# ---------- 画图+日志 ----------
def plot_and_log(sim, fams, Z, out_prefix, use_kv):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(max(8, len(fams) * 0.15), 6))
    im = plt.imshow(sim, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(im, label='Cosine Similarity')
    plt.xticks(range(len(fams)), fams, rotation=90)
    plt.yticks(range(len(fams)), fams)
    mode = "KV" if use_kv else "Key"
    plt.title(f"{mode} Hash Similarity")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=300)
    plt.close()
    with open(f"{out_prefix}.log", 'w', encoding='utf-8') as log:
        for idx, (fam, vec) in enumerate(zip(fams, Z)):
            log.write(f"{idx}\t{fam}\t" + " ".join(f"{v:.6f}" for v in vec) + "\n")
    print(f"✅ 热图：{out_prefix}.png")
    print(f"✅ 日志：{out_prefix}.log")

# ---------- 入口 ----------
if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument("--data",  help="训练用 json 列表（训练模式）")
    ps.add_argument("--out",   help="输出前缀（训练模式）")
    ps.add_argument("--kv",    action="store_true", help="同时 hash 键值（默认只键名）")
    ps.add_argument("--db",    help="库前缀（匹配模式）")
    ps.add_argument("--match", help="新样本 json 文件（匹配模式），支持单条或列表")
    ps.add_argument("--top",   type=int, default=3, help="每个样本返回前 K 个（默认3）")
    ps.add_argument("--fewshot-out", help="Few-shot输出文件路径（纯列表格式）")
    ps.add_argument("--train-data", help="原始训练数据文件路径（匹配模式专用）")
    args = ps.parse_args()

    if args.data and args.out:
        build_and_save(args.data, args.out, args.kv)
    elif args.db and args.match:
        match_single(args.db, args.match, args.top, args.kv, args.fewshot_out, args.train_data)
    else:
        ps.print_help()

# # 分文件召回存储
# #!/usr/bin/env python3
# # sim_heatmap_kv_switch.py  训练+热图+日志+匹配+Few-shot输出
# import json, mmh3, numpy as np, matplotlib.pyplot as plt, argparse, tqdm, copy, os
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------- 向量化 ----------
# def ioc2vec(ioc: dict, use_kv: bool = False) -> np.ndarray:
#     z = [0.0] * 32
#     for k, v in sorted(ioc.items()):
#         z[mmh3.hash(k) % 32] += 1.0
#         if use_kv:
#             z[mmh3.hash(str(v)) % 32] += 1.0
#     z = np.log1p(z) / 10.0
#     return np.array(z, dtype='float32')

# # ---------- 找 label ----------
# def find_label(item: dict) -> str:
#     if "json" in item and "label" in item["json"]:
#         return item["json"]["label"]["official"]
#     if "label" in item:
#         return item["label"]["official"]
#     raise KeyError("no label.official")

# # ---------- 找 IOC ----------
# def find_ioc(item: dict) -> dict:
#     ioc = None

#     if "json" in item:
#         if isinstance(item["json"], dict) and "json" in item["json"]:
#             ioc = item["json"]["json"]
#         else:
#             ioc = item["json"]
#     else:
#         ioc = item

#     if isinstance(ioc, list):
#         if len(ioc) == 0:
#             raise KeyError("IOC list is empty")
#         ioc = ioc[0]

#     if not isinstance(ioc, dict):
#         raise KeyError(f"IOC is not a dict, got {type(ioc)}")

#     return ioc

# # ---------- 训练+保存 ----------
# def build_and_save(data_path, out_prefix, use_kv):
#     with open(data_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     vecs, fams, raw_items = [], [], []
#     for item in tqdm.tqdm(data, desc="training"):
#         try:
#             fam = find_label(item)
#             ioc = find_ioc(item)
#         except KeyError:
#             continue
#         vecs.append(ioc2vec(ioc, use_kv))
#         fams.append(fam)
#         raw_items.append(copy.deepcopy(item))

#     Z = np.stack(vecs)
#     suffix = "_kv" if use_kv else "_key"
#     np.save(f"{out_prefix}{suffix}_Z.npy", Z)
#     json.dump(fams, open(f"{out_prefix}{suffix}_fams.json", 'w', encoding='utf-8'))
#     json.dump(raw_items, open(f"{out_prefix}{suffix}_raw.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
#     print(f"✅ 向量库已保存：{out_prefix}{suffix}_Z.npy")
#     print(f"✅ 标签库已保存：{out_prefix}{suffix}_fams.json")
#     print(f"✅ 原始数据已保存：{out_prefix}{suffix}_raw.json")

#     sim = cosine_similarity(Z)
#     plot_and_log(sim, fams, Z, out_prefix, use_kv)
#     return Z, fams, raw_items

# # ---------- 匹配 + Few-shot输出（支持多样本批量处理） ----------
# def match_single(db_prefix, new_path, top_k, use_kv, fewshot_out=None, train_data_path=None):
#     suffix = "_kv" if use_kv else "_key"
#     Z_db = np.load(f"{db_prefix}{suffix}_Z.npy")
#     fams = json.load(open(f"{db_prefix}{suffix}_fams.json", encoding='utf-8'))

#     raw_items = None
#     try:
#         raw_items = json.load(open(f"{db_prefix}{suffix}_raw.json", encoding='utf-8'))
#     except FileNotFoundError:
#         pass

#     if raw_items is None and train_data_path:
#         try:
#             with open(train_data_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             raw_items = []
#             for item in data:
#                 try:
#                     _ = find_label(item)
#                     _ = find_ioc(item)
#                     raw_items.append(copy.deepcopy(item))
#                 except KeyError:
#                     continue
#         except Exception as e:
#             print(f"⚠️ 从训练数据加载失败：{e}")

#     if raw_items is None:
#         print("⚠️ 警告：未找到原始数据，Few-shot输出将不可用。")

#     # 加载未知攻击库（支持单条或列表）
#     with open(new_path, encoding='utf-8') as f:
#         new_data = json.load(f)
    
#     # 统一转为列表处理
#     if not isinstance(new_data, list):
#         new_data = [new_data]
    
#     print(f"\n📦 未知攻击库包含 {len(new_data)} 个样本，开始批量匹配...")
    
#     all_results = []
    
#     for idx, new_item in enumerate(new_data):
#         print(f"\n{'='*50}")
#         print(f"🔍 样本 [{idx+1}/{len(new_data)}]")
        
#         try:
#             z_new = ioc2vec(find_ioc(new_item), use_kv).reshape(1, -1)
#         except KeyError as e:
#             print(f"  ❌ 跳过：无法提取IOC ({e})")
#             continue
            
#         sims = cosine_similarity(z_new, Z_db)[0]
#         top_idx = np.argsort(sims)[::-1][:top_k]

#         print(f"  Top-{top_k} 最相似样本：")
#         for rank, tidx in enumerate(top_idx, 1):
#             print(f"    [{rank}] cos={sims[tidx]:.3f}  official={fams[tidx]}")

#         # 构建当前样本的few-shot结果
#         if raw_items:
#             fewshot_list = [raw_items[tidx] for tidx in top_idx if tidx < len(raw_items)]
            
#             # 确定输出路径
#             if fewshot_out:
#                 # 多样本时自动编号
#                 if len(new_data) > 1:
#                     base, ext = os.path.splitext(fewshot_out)
#                     out_file = f"{base}_sample{idx+1}{ext}"
#                 else:
#                     out_file = fewshot_out
                    
#                 with open(out_file, 'w', encoding='utf-8') as f:
#                     json.dump(fewshot_list, f, ensure_ascii=False, indent=2)
#                 print(f"  ✅ Few-shot列表已保存至：{out_file}")
#             else:
#                 print(f"  📋 Few-shot JSON列表（{len(fewshot_list)}个）：")
#                 print(json.dumps(fewshot_list, ensure_ascii=False, indent=2))
            
#             all_results.append({
#                 "sample_idx": idx,
#                 "top_k": top_k,
#                 "matches": [{"idx": int(tidx), "cos": float(sims[tidx]), "official": fams[tidx]} for tidx in top_idx],
#                 "fewshot_data": fewshot_list
#             })
#         else:
#             all_results.append({
#                 "sample_idx": idx,
#                 "top_k": top_k,
#                 "matches": [{"idx": int(tidx), "cos": float(sims[tidx]), "official": fams[tidx]} for tidx in top_idx]
#             })
    
#     print(f"\n{'='*50}")
#     print(f"✅ 全部完成！共处理 {len(new_data)} 个样本")
#     return all_results

# # ---------- 画图+日志 ----------
# def plot_and_log(sim, fams, Z, out_prefix, use_kv):
#     plt.figure(figsize=(max(8, len(fams) * 0.15), 6))
#     im = plt.imshow(sim, cmap='coolwarm', vmin=0, vmax=1)
#     plt.colorbar(im, label='Cosine Similarity')
#     plt.xticks(range(len(fams)), fams, rotation=90)
#     plt.yticks(range(len(fams)), fams)
#     mode = "KV" if use_kv else "Key"
#     plt.title(f"{mode} Hash Similarity")
#     plt.tight_layout()
#     plt.savefig(f"{out_prefix}.png", dpi=300)
#     plt.close()
#     with open(f"{out_prefix}.log", 'w', encoding='utf-8') as log:
#         for idx, (fam, vec) in enumerate(zip(fams, Z)):
#             log.write(f"{idx}\t{fam}\t" + " ".join(f"{v:.6f}" for v in vec) + "\n")
#     print(f"✅ 热图：{out_prefix}.png")
#     print(f"✅ 日志：{out_prefix}.log")

# # ---------- 入口 ----------
# if __name__ == "__main__":
#     ps = argparse.ArgumentParser()
#     ps.add_argument("--data",  help="训练用 json 列表（训练模式）")
#     ps.add_argument("--out",   help="输出前缀（训练模式）")
#     ps.add_argument("--kv",    action="store_true", help="同时 hash 键值（默认只键名）")
#     ps.add_argument("--db",    help="库前缀（匹配模式）")
#     ps.add_argument("--match", help="新样本 json 文件（匹配模式），支持单条或列表")
#     ps.add_argument("--top",   type=int, default=3, help="每个样本返回前 K 个（默认3）")
#     ps.add_argument("--fewshot-out", help="Few-shot输出文件路径（可选，多样本时自动编号）")
#     ps.add_argument("--train-data", help="原始训练数据文件路径（匹配模式专用）")
#     args = ps.parse_args()

#     if args.data and args.out:
#         build_and_save(args.data, args.out, args.kv)
#     elif args.db and args.match:
#         match_single(args.db, args.match, args.top, args.kv, args.fewshot_out, args.train_data)
#     else:
#         ps.print_help()

# 原始方法
# #!/usr/bin/env python3
# # sim_heatmap_kv_switch.py  训练+热图+日志+匹配+Few-shot输出
# import json, mmh3, numpy as np, matplotlib.pyplot as plt, argparse, tqdm, copy
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------- 向量化 ----------
# def ioc2vec(ioc: dict, use_kv: bool = False) -> np.ndarray:
#     z = [0.0] * 32
#     for k, v in sorted(ioc.items()):
#         z[mmh3.hash(k) % 32] += 1.0
#         if use_kv:
#             z[mmh3.hash(str(v)) % 32] += 1.0
#     z = np.log1p(z) / 10.0
#     return np.array(z, dtype='float32')

# # ---------- 找 label ----------
# def find_label(item: dict) -> str:
#     if "json" in item and "label" in item["json"]:
#         return item["json"]["label"]["official"]
#     if "label" in item:
#         return item["label"]["official"]
#     raise KeyError("no label.official")

# # ---------- 找 IOC ----------
# def find_ioc(item: dict) -> dict:
#     ioc = None

#     if "json" in item:
#         if isinstance(item["json"], dict) and "json" in item["json"]:
#             ioc = item["json"]["json"]
#         else:
#             ioc = item["json"]
#     else:
#         ioc = item

#     if isinstance(ioc, list):
#         if len(ioc) == 0:
#             raise KeyError("IOC list is empty")
#         ioc = ioc[0]

#     if not isinstance(ioc, dict):
#         raise KeyError(f"IOC is not a dict, got {type(ioc)}")

#     return ioc

# # ---------- 训练+保存 ----------
# def build_and_save(data_path, out_prefix, use_kv):
#     with open(data_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     vecs, fams, raw_items = [], [], []
#     for item in tqdm.tqdm(data, desc="training"):
#         try:
#             fam = find_label(item)
#             ioc = find_ioc(item)
#         except KeyError:
#             continue
#         vecs.append(ioc2vec(ioc, use_kv))
#         fams.append(fam)
#         raw_items.append(copy.deepcopy(item))

#     Z = np.stack(vecs)
#     suffix = "_kv" if use_kv else "_key"
#     np.save(f"{out_prefix}{suffix}_Z.npy", Z)
#     json.dump(fams, open(f"{out_prefix}{suffix}_fams.json", 'w', encoding='utf-8'))
#     json.dump(raw_items, open(f"{out_prefix}{suffix}_raw.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
#     print(f"✅ 向量库已保存：{out_prefix}{suffix}_Z.npy")
#     print(f"✅ 标签库已保存：{out_prefix}{suffix}_fams.json")
#     print(f"✅ 原始数据已保存：{out_prefix}{suffix}_raw.json")

#     sim = cosine_similarity(Z)
#     plot_and_log(sim, fams, Z, out_prefix, use_kv)
#     return Z, fams, raw_items

# # ---------- 匹配 + Few-shot输出 ----------
# def match_single(db_prefix, new_path, top_k, use_kv, fewshot_out=None, train_data_path=None):
#     suffix = "_kv" if use_kv else "_key"
#     Z_db = np.load(f"{db_prefix}{suffix}_Z.npy")
#     fams = json.load(open(f"{db_prefix}{suffix}_fams.json", encoding='utf-8'))

#     raw_items = None
#     try:
#         raw_items = json.load(open(f"{db_prefix}{suffix}_raw.json", encoding='utf-8'))
#     except FileNotFoundError:
#         pass

#     if raw_items is None and train_data_path:
#         try:
#             with open(train_data_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             raw_items = []
#             for item in data:
#                 try:
#                     _ = find_label(item)
#                     _ = find_ioc(item)
#                     raw_items.append(copy.deepcopy(item))
#                 except KeyError:
#                     continue
#         except Exception as e:
#             print(f"⚠️ 从训练数据加载失败：{e}")

#     if raw_items is None:
#         print("⚠️ 警告：未找到原始数据，Few-shot输出将不可用。")

#     with open(new_path, encoding='utf-8') as f:
#         new_item = json.load(f)
#     new_item = new_item[0] if isinstance(new_item, list) else new_item
#     z_new = ioc2vec(find_ioc(new_item), use_kv).reshape(1, -1)
#     sims = cosine_similarity(z_new, Z_db)[0]
#     top_idx = np.argsort(sims)[::-1][:top_k]

#     print(f"\n🔍 Top-{top_k} 最相似样本：")
#     for rank, idx in enumerate(top_idx, 1):
#         print(f"  [{rank}] cos={sims[idx]:.3f}  official={fams[idx]}")

#     # 输出：直接是3个JSON对象的列表，无其他包装
#     if raw_items:
#         fewshot_list = [raw_items[idx] for idx in top_idx if idx < len(raw_items)]

#         if fewshot_out:
#             with open(fewshot_out, 'w', encoding='utf-8') as f:
#                 json.dump(fewshot_list, f, ensure_ascii=False, indent=2)
#             print(f"\n✅ Few-shot列表（{len(fewshot_list)}个）已保存至：{fewshot_out}")
#         else:
#             print(f"\n📋 Few-shot JSON列表：")
#             print(json.dumps(fewshot_list, ensure_ascii=False, indent=2))

#         return fewshot_list
#     return None

# # ---------- 画图+日志 ----------
# def plot_and_log(sim, fams, Z, out_prefix, use_kv):
#     plt.figure(figsize=(max(8, len(fams) * 0.15), 6))
#     im = plt.imshow(sim, cmap='coolwarm', vmin=0, vmax=1)
#     plt.colorbar(im, label='Cosine Similarity')
#     plt.xticks(range(len(fams)), fams, rotation=90)
#     plt.yticks(range(len(fams)), fams)
#     mode = "KV" if use_kv else "Key"
#     plt.title(f"{mode} Hash Similarity")
#     plt.tight_layout()
#     plt.savefig(f"{out_prefix}.png", dpi=300)
#     plt.close()
#     with open(f"{out_prefix}.log", 'w', encoding='utf-8') as log:
#         for idx, (fam, vec) in enumerate(zip(fams, Z)):
#             log.write(f"{idx}\t{fam}\t" + " ".join(f"{v:.6f}" for v in vec) + "\n")
#     print(f"✅ 热图：{out_prefix}.png")
#     print(f"✅ 日志：{out_prefix}.log")

# # ---------- 入口 ----------
# if __name__ == "__main__":
#     ps = argparse.ArgumentParser()
#     ps.add_argument("--data",  help="训练用 json 列表（训练模式）")
#     ps.add_argument("--out",   help="输出前缀（训练模式）")
#     ps.add_argument("--kv",    action="store_true", help="同时 hash 键值（默认只键名）")
#     ps.add_argument("--db",    help="库前缀（匹配模式）")
#     ps.add_argument("--match", help="新样本 json 文件（匹配模式）")
#     ps.add_argument("--top",   type=int, default=3, help="返回前 K 个（默认3）")
#     ps.add_argument("--fewshot-out", help="Few-shot输出文件路径（可选）")
#     ps.add_argument("--train-data", help="原始训练数据文件路径（匹配模式专用）")
#     args = ps.parse_args()

#     if args.data and args.out:
#         build_and_save(args.data, args.out, args.kv)
#     elif args.db and args.match:
#         match_single(args.db, args.match, args.top, args.kv, args.fewshot_out, args.train_data)
#     else:
#         ps.print_help()
