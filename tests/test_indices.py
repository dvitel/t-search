from functools import partial
from itertools import product
import math
import torch

from t_search.spatial import RTreeIndex, InteractionIndex, SpearmanCorIndex, RCosIndex, GridIndex
from t_search.spatial import SpatialIndex, VectorStorage


def test_storage(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cpu"):
    storage = VectorStorage(capacity, dims, dtype=dtype, device=device)
    all_ids = []
    all_chunks = []
    for chunk_sz in range(1, 10):
        chunk = torch.randn((chunk_sz, dims), dtype=dtype, device=device)
        ids = storage._alloc_vectors(chunk)
        all_ids.append(ids)
        all_chunks.append(chunk)
    assert storage.cur_id == sum(len(l) for l in all_ids), "Storage did not allocate all vectors correctly."
    assert list(range(storage.cur_id)) == [el for l in all_ids for el in l], "Storage allocated vectors with wrong ids."
    chunk_tensor = torch.cat(all_chunks, dim=0)
    assert torch.equal(storage.get_vectors(None), chunk_tensor), "Storage did not return all vectors"
    selected_ids = [el for l in all_ids[1:-1] for el in l]
    selected_vectors = storage.get_vectors(selected_ids)
    chunk_tensor2 = torch.cat(all_chunks[1:-1], dim=0)
    assert torch.equal(selected_vectors, chunk_tensor2), "Storage did not return correct vectors for selected ids."
    chunk_id = 5
    assert torch.equal(storage.get_vectors((all_ids[chunk_id][0], all_ids[chunk_id][-1] + 1)), all_chunks[chunk_id]), "Storage did not return correct vector for id 0."
    el_ids = [1, 5, 42]
    query = storage.get_vectors(el_ids)
    close_ids = storage.find_close(None, query)
    for close_id, el_id in zip(close_ids, el_ids):
        if close_id != el_id:
            close_vector = storage.get_vectors(close_id)
            el_vector = storage.get_vectors(el_id)
            assert torch.allclose(close_vector, el_vector, rtol=storage.rtol, atol=storage.atol), "Storage did not return close vectors correctly."
    mbr = storage.find_mbr(None)
    id_range = (13, 31)
    in_range = storage.find_in_range(id_range, mbr)
    # all_vectors = storage.get_vectors(None)
    assert in_range == list(range(*id_range)), "Storage did not return all"

    pass

import time
def test_spatial_index(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=4096, query_batch_size=256, dim_batch_size=4,
                        num_points_per_group = 100, num_groups = 100,
                        min_r = 0.3, max_r = 0.7, time_deltas = False,
                        index = SpatialIndex):
    
    idx = index(capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size)
    all_ids_grouped = []
    all_ids = []
    all_points = []
    start_time = time.time()
    times = []
    for i in range(num_groups):
        pi = torch.rand((num_points_per_group, dims), dtype=dtype, device=device)
        new_ids = idx.insert(pi)
        all_ids.extend(new_ids)
        all_ids_grouped.append(new_ids)
        all_points.extend(pi.tolist())
        iter_end = time.time()
        total_duration = iter_end - start_time
        if time_deltas:
            start_time = iter_end
        times.append(total_duration)
        print(f"Inserted group {i + 1:02}/{num_groups} in {total_duration:06.2f} seconds, "
              f"total {idx.cur_id:06} points, {store_batch_size:04}:{query_batch_size:03}:{dim_batch_size:03}")
    search_points = torch.randint(idx.cur_id, (100,)).tolist()
    search_tensors = idx.get_vectors(search_points)
    found_ids = idx.find_close(None, search_tensors)
    # should_be_ids = [all_ids[sp] for sp in search_points]
    assert sorted(search_points) == sorted(found_ids), "Spatial index did not return correct ids for search points."
    # range_query = torch.full((2, dims), 0, dtype=dtype, device=device)
    # range_query[0] = min_r
    # range_query[1] = max_r
    # found_ids = idx.find_in_range(None, range_query)
    # expected_ids = [el for i in range(int(min_r / step), int(max_r / step) + 1) for el in all_ids_grouped[i]]
    # assert sorted(found_ids) == sorted(expected_ids), "Spatial index did not return correct ids in range"

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."
    
def test_spatial_index_query(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=4096, query_batch_size=256, dim_batch_size=4,
                        num_points_per_group = 100, num_groups = 101, index = SpatialIndex,
                        max_children = 64):
    
    idx = index(capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size, max_children = max_children)
    # start_insert_time = time.time()
    # idx.insert(torch.rand((capacity, dims), dtype = dtype, device=device))
    # for bin_id, point_ids in idx.bins.items():
    #     point_tensor = idx.get_vectors(point_ids)
    #     bin_tensor = torch.tensor(bin_id, dtype=idx.epsilons.dtype, device=idx.epsilons.device)
    #     bin_min = bin_tensor * idx.epsilons
    #     bin_max = bin_min + idx.epsilons
    #     assert torch.all((point_tensor >= bin_min) & (point_tensor <= bin_max)), "Grid index did not insert points in correct bins."    
    #     pass
    # print(f"Inserted {idx.cur_id:06} points in {start_time - start_insert_time:06.2f} seconds")
    insert_time = 0
    query_time = 0
    times = []
    for i in range(num_groups):
        # ids_range = (0, (i + 1) * num_points_per_group)
        new_tensors = ((i * 0.25) % 1) + 0.25 * torch.rand((num_points_per_group, dims), dtype = dtype, device=device)
        start_insert_time = time.time()
        idx.insert(new_tensors)        
        end_insert_time = time.time()
        orig_ids = torch.randint(idx.cur_id, (num_points_per_group,)).tolist()
        orig_ids.sort()
        query_points = idx.get_vectors(orig_ids)
        start_query_time = time.time()
        found_ids = idx.query_points(query_points)
        end_query_time = time.time()
        # found_ids = []
        # for query_point in query_points:
        #     one_found_ids = idx.query_points(query_point.unsqueeze(0)) # query one point at a time
        #     found_ids.extend(one_found_ids)
        found_ids.sort()
        assert found_ids == orig_ids, f"Spatial index did not return correct ids for group"
        insert_time += end_insert_time - start_insert_time
        query_time += end_query_time - start_query_time
        total_time = insert_time + query_time

        times.append(total_time)
        if i % 10 == 0:
            print(f"Queried group {i + 1:03}/{num_groups} in {total_time:05.2f} [{insert_time:05.2f},{query_time:05.2f}] seconds, "
                f"total {idx.cur_id:06} points, {store_batch_size:04}:{query_batch_size:03}:{dim_batch_size:03}|{max_children:03}")

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."


def test_spatial_index_query2(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        num_points_per_group = 100, num_groups = 101, indices: dict = {"spatial": SpatialIndex}):
    
    import numpy as np
    
    idxs = {idx_name: index(capacity=capacity, dims=dims, dtype=dtype, device=device) 
                        for idx_name, index in indices.items()}
    insert_times = {n:0 for n in idxs.keys()}
    query_times = {n:0 for n in idxs.keys()}
    range_times = {n:0 for n in idxs.keys()}
    times = {n:[] for n in idxs.keys()}
    cur_id = 0
    for i in range(num_groups):
        cur_num_points = np.random.randint(num_points_per_group // 2, num_points_per_group + 1)
        means = torch.rand(dims, dtype=dtype, device=device)
        std = np.random.rand() * 0.05 + 0.025
        stds = torch.full((cur_num_points, dims), std, dtype=dtype, device=device)
        distr = torch.normal(mean = means, std = stds)   
        distr.clamp_(0, 1)     
        same_count = min(cur_id, np.random.randint(0, distr.shape[0]))
        selected_ids = None
        if same_count > 0:
            tmp_perm = torch.randperm(distr.shape[0], device=device)
            selected_ids = tmp_perm[:same_count].tolist()
            del tmp_perm
            tmp_perm = torch.randperm(cur_id, device=device)
            from_idx_ids = tmp_perm[:same_count].tolist()
            del tmp_perm
        new_cur_ids = []
        idxs_returned_ids = []
        for idx_name, idx in idxs.items():
            if not idx_name.startswith('rtree'):
                assert idx.cur_id == cur_id
            if selected_ids is not None:
                distr[selected_ids] = idx.vectors[from_idx_ids]
            start_insert_time = time.time()
            returned_ids = idx.insert(distr)   
            end_insert_time = time.time()
            idxs_returned_ids.append(returned_ids)     
            insert_time = end_insert_time - start_insert_time
            insert_times[idx_name] += insert_time
            if not idx_name.startswith('rtree'):
                new_cur_ids.append(idx.cur_id)
        assert np.all([x == new_cur_ids[0] for x in new_cur_ids])
        cur_id = new_cur_ids[0]

        #querying points
        cur_num_points2 = np.random.randint(1, num_points_per_group + 1)
        same_count = min(cur_id, np.random.randint(0, cur_num_points2))
        queries = torch.rand((cur_num_points2, dims), dtype=dtype, device=device)
        tmp_perm = torch.randperm(queries.shape[0], device=device)
        selected_ids = tmp_perm[:same_count].tolist()
        del tmp_perm
        tmp_perm = torch.randperm(cur_id, device=device)
        to_ids = tmp_perm[:same_count].tolist()
        del tmp_perm
        for idx_name, idx in idxs.items():
            idx_q = queries.clone()
            idx_q[selected_ids] = idx.get_vectors(to_ids)
            # assert idx.cur_id == cur_id
            start_query_time = time.time()
            found_ids = idx.query_points(idx_q)
            assert len(found_ids) == idx_q.shape[0]
            end_query_time = time.time()
            query_time = end_query_time - start_query_time
            query_times[idx_name] += query_time
            found_ids_set = set(found_ids)
            to_ids_set = set(to_ids)
            to_ids_set.add(-1)
            if not idx_name.startswith('rtree'):
                assert set.issubset(to_ids_set, found_ids_set), f"Spatial index {idx_name} did not return correct ids for group {i + 1:02}/{num_groups}"
            del idx_q

        for idx_name, idx in idxs.items():
            total_time = insert_times[idx_name] + query_times[idx_name]
            times[idx_name].append(total_time)
        if i % 10 == 0:
            print(f"{i + 1:03}/{num_groups}")
            for idx_name, idx in sorted(idxs.items(), key=lambda x: times[x[0]][-1], reverse=True):
                print(f"\t{idx_name:<10} in {times[idx_name][-1]:05.2f} [{insert_times[idx_name]:05.2f},{query_times[idx_name]:05.2f}] seconds, "
                    f"total {cur_id:06} points")

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."


def test_spatial_index_range(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=4096, query_batch_size=256, dim_batch_size=4,
                        num_points_per_group = 100, num_groups = 1000,
                        time_deltas = False, index = SpatialIndex):
    
    idx = index(capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size)
    # idx.insert(torch.rand((capacity, dims), dtype = dtype, device=device))
    # for bin_id, point_ids in idx.bins.items():
    #     point_tensor = idx.get_vectors(point_ids)
    #     bin_tensor = torch.tensor(bin_id, dtype=idx.epsilons.dtype, device=idx.epsilons.device)
    #     bin_min = bin_tensor * idx.epsilons
    #     bin_max = bin_min + idx.epsilons
    #     assert torch.all((point_tensor >= bin_min) & (point_tensor <= bin_max)), "Grid index did not insert points in correct bins."    
    #     pass
    start_time = time.time()
    times = []
    for i in range(num_groups):
        # ids_range = (0, (i + 1) * num_points_per_group)
        new_tensors = ((i * 0.25) % 1) + 0.25 * torch.rand((num_points_per_group, dims), dtype = dtype, device=device)
        new_ids = idx.insert(new_tensors)        
        min_p = new_tensors.min(dim=0).values
        max_p = new_tensors.max(dim=0).values
        qrange = torch.stack((min_p, max_p), dim=0) # (2, dims)
        new_ids = set(new_ids)
        found_ids = idx.query_range(qrange)
        found_ids = set(found_ids)
        assert set.issubset(new_ids, found_ids), f"Spatial index did not return correct ids for group {i + 1:02}/{num_groups}"
        iter_end = time.time()
        total_duration = iter_end - start_time
        if time_deltas:
            start_time = iter_end
        times.append(total_duration)
        if i % 10 == 0:
            print(f"Queried group {i + 1:02}/{num_groups} in {total_duration:06.2f} seconds, "
                f"{store_batch_size:04}:{query_batch_size:03}:{dim_batch_size:03}")

    return times
    # assert idx.cur_id == num_points_per_group * num_groups, "Spatial index did not allocate all vectors correctly."


def test_time(f, **arg_combs):    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    keys = list(arg_combs.keys())
    for arg_comb in product(*arg_combs.values()):
        times = f(**{k: v for k, v in zip(keys, arg_comb)})
        plt.plot(times, label=f"({arg_comb})")
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_time2(f, *arg_combs):    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    for arg_comb in arg_combs:
        name = arg_comb.pop("_name")
        print(f"--- {name} ---")
        times = f(**arg_comb)
        plt.plot(times, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()    

def test_time_all(f, **kwargs):    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    times = f(**kwargs)
    for time_name, xy in times.items():
        plt.plot(xy, label=time_name)
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()        

def test_grid_index(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    idx = GridIndex(epsilons=1, max_children = 2000, capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids1 = idx.insert(p)
    p_ids2 = idx.insert(p)
    assert sorted(p_ids1) == sorted(p_ids2), "Grid index did not return same ids for same points."
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids3 = idx.insert(p) # should trigger rebuild
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids4 = idx.insert(p)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids5 = idx.insert(p)

    v = idx.get_vectors(p_ids1)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids1), "Grid index did not return correct"
    v = idx.get_vectors(p_ids2)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids2), "Grid index did not return correct"
    v = idx.get_vectors(p_ids3)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids3), "Grid index did not return correct"
    v = idx.get_vectors(p_ids4)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids4), "Grid index did not return correct"
    
    r_ids = idx.query_range(torch.tensor([[0] * dims, [1] * dims], dtype=dtype, device=device))
    assert sorted(r_ids) == list(range(idx.cur_id)), "Grid index did not return all ids in range query."

    
    pass 

def test_rcos_index(capacity = 100_000, dims = 1024, dtype = torch.float32, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    target = torch.full((dims,), 0.5, dtype=dtype, device=device)
    idx = RCosIndex(target, max_children = 10, capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids1 = idx.insert(p)
    p_ids2 = idx.insert(p)
    assert sorted(p_ids1) == sorted(p_ids2), "Grid index did not return same ids for same points."
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids3 = idx.insert(p) # should trigger rebuild
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids4 = idx.insert(p)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids5 = idx.insert(p)

    v = idx.get_vectors(p_ids1)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids1), "Grid index did not return correct"
    v = idx.get_vectors(p_ids2)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids2), "Grid index did not return correct"
    v = idx.get_vectors(p_ids3)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids3), "Grid index did not return correct"
    v = idx.get_vectors(p_ids4)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids4), "Grid index did not return correct"
    
    qry0 = torch.ones_like(target, dtype=dtype, device=device)
    qry0[0] = 0
    qry1 = torch.zeros_like(target, dtype=dtype, device=device)
    qry1[0] = 1
    qry = torch.stack((qry0, qry1), dim=0) # (2, dims)
    r_ids = idx.query_range(qry)
    assert sorted(r_ids) == list(range(idx.cur_id)), "Grid index did not return all ids in range query."

    
    pass 

def test_scor_index(capacity = 100_000, dims = 1024, dtype = torch.float32, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    target = torch.full((dims,), 0.5, dtype=dtype, device=device)
    idx = SpearmanCorIndex(target, max_children = 1000, capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids1 = idx.insert(p)
    p_ids2 = idx.insert(p)
    assert sorted(p_ids1) == sorted(p_ids2), "Grid index did not return same ids for same points."
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids3 = idx.insert(p) # should trigger rebuild
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids4 = idx.insert(p)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids5 = idx.insert(p)

    v = idx.get_vectors(p_ids1)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids1), "Grid index did not return correct"
    v = idx.get_vectors(p_ids2)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids2), "Grid index did not return correct"
    v = idx.get_vectors(p_ids3)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids3), "Grid index did not return correct"
    v = idx.get_vectors(p_ids4)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids4), "Grid index did not return correct"
    
    qry = torch.tensor([[0], [2]], dtype=dtype, device=device)
    r_ids = idx.query_range(qry)
    assert sorted(r_ids) == list(range(idx.cur_id)), "Grid index did not return all ids in range query."

    
    pass 

def test_int_index(capacity = 100_000, dims = 1024, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8):
    target = torch.full((dims,), 0.5, dtype=dtype, device=device)
    idx = InteractionIndex(target, max_children = 10, capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids1 = idx.insert(p)
    print("Insert 1 done")
    p_ids2 = idx.insert(p)
    print("Insert 2 done")
    assert sorted(p_ids1) == sorted(p_ids2), "Grid index did not return same ids for same points."
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids3 = idx.insert(p) # should trigger rebuild
    print("Insert 3 done")
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids4 = idx.insert(p)
    print("Insert 4 done")
    p = torch.rand((1000, dims), dtype=dtype, device=device)
    p_ids5 = idx.insert(p)
    print("Insert 5 done")

    v = idx.get_vectors(p_ids1)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids1), "Grid index did not return correct"
    v = idx.get_vectors(p_ids2)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids2), "Grid index did not return correct"
    v = idx.get_vectors(p_ids3)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids3), "Grid index did not return correct"
    v = idx.get_vectors(p_ids4)
    q_ids = idx.query_points(v)
    assert sorted(q_ids) == sorted(p_ids4), "Grid index did not return correct"
    
    r_ids = idx.query_range(torch.ones((dims,), dtype=dtype, device=device))
    assert sorted(r_ids) == list(range(idx.cur_id)), "Grid index did not return all ids in range query."

    
    pass 

# visualize_2d_frame_id = 0

def visualize_2d(x, y, rects=None, epsilons=None, xrange = None, yrange = None):
    """
    Visualizes scattered points, rectangles, and an optional grid.

    Args:
        x (list[float]): x-coordinates of scattered points.
        y (list[float]): y-coordinates of scattered points.
        rects (list[tuple[float, float, float, float]]): List of rectangles, each defined as (x_min, y_min, width, height).
        epsilons (tuple[float, float]): Grid spacing for x and y axes (optional).
    """
    # global visualize_2d_frame_id
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.ticker as ticker
    import numpy as np
    plt.ion()
    plt.clf()
    # plt.figure(figsize=(10, 6))
    
    # Plot scattered points
    plt.scatter(x, y, color="black", s = 4)
    
    # Plot grid
    if epsilons:
        x_epsilon, y_epsilon = epsilons
        plt.gca().set_xticks(np.arange(math.floor(xrange[0] / x_epsilon), math.floor(xrange[1] / x_epsilon) + 1) * x_epsilon)
        plt.gca().set_yticks(np.arange(math.floor(yrange[0] / y_epsilon), math.floor(yrange[1] / y_epsilon) + 1) * y_epsilon)
        plt.gca().set_xticklabels([])
        plt.gca().set_xticklabels([])
        plt.gca().tick_params(labelbottom=False, labelleft=False)
        plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    
    # Plot rectangles
    if rects:
        for rect in rects:
            x_min, y_min, width, height = rect
            rect_patch = patches.Rectangle((x_min, y_min), width, height, 
                                           linewidth=1, edgecolor="red", facecolor="red", alpha=0.2)
            plt.gca().add_patch(rect_patch)
    
    # Add labels and legend
    # plt.xlabel("X")
    # plt.ylabel("Y")
    if xrange is not None:
        plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.02f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.02f'))
    # plt.title("2D Visualization of Points and Rectangles")
    # plt.legend()
    plt.tight_layout()
    # plt.ioff()
    # plt.show()
    # if visualize_2d_frame_id == 0:
    #     plt.get_current_fig_manager().full_screen_toggle()
    # visualize_2d_frame_id += 1
    plt.pause(2)

# Example usage
# x = [1, 2, 3, 4, 5]
# y = [5, 4, 3, 2, 1]
# rects = [(1.5, 1.5, 2, 2), (3.5, 3.5, 1, 1)]
# epsilons = (1, 1)

# visualize_2d(x, y, rects=rects, epsilons=epsilons)
# pass

def viz_idx(idx: SpatialIndex):
    if hasattr(idx, 'epsilons'):
        epsilonx, epsilony = idx.epsilons.tolist()
    else:
        epsilonx, epsilony = 0.1, 0.1
    vectors = idx.get_vectors(None)
    if isinstance(idx, GridIndex):
        rects = [(bin_id_x * epsilonx, bin_id_y * epsilony, epsilonx, epsilony) for bin_id_x, bin_id_y in idx.bins.keys()]
    elif isinstance(idx, RTreeIndex):
        mbrs_list = idx.get_mbrs()
        rects = [(mbr[0,0].item(), mbr[0,1].item(), w[0].item(), w[1].item()) for mbr in mbrs_list for w in [mbr[1] - mbr[0]]]
    xs = vectors[:, 0].tolist()
    ys = vectors[:, 1].tolist()
    visualize_2d(xs, ys,
                 rects=rects,
                 epsilons=(epsilonx, epsilony),
                 xrange=(0, 1), yrange=(0, 1))
    
def test_idx_distr(capacity = 100_000, dims = 2, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8,
                        num_groups = 20, group_size=10, idxb = partial(GridIndex, epsilons=1, max_bin_size = 10)):
    idx = idxb(capacity=capacity, dims=dims, dtype=dtype, device=device,
                    store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                    dim_batch_size = dim_batch_size)
    for num_groups in range(num_groups):
        means = torch.rand(dims, dtype=dtype, device=device)
        stds = torch.full((group_size, dims), 0.05, dtype=dtype, device=device)
        distr = torch.normal(mean = means, std = stds)
        distr.clamp_(0, 1)
        # distr = shifts + torch.rand((100, dims), dtype=dtype, device=device) * 0.1
        idx.insert(distr)
        viz_idx(idx)    

    pass

def test_rtree_index(capacity = 100_000, dims = 2, dtype = torch.float16, device = "cuda",
                        store_batch_size=1024, query_batch_size=256, dim_batch_size=8,
                        num_points_per_group = 3, num_groups = 100):
    idx = RTreeIndex(max_children = 10, capacity=capacity, dims=dims, dtype=dtype, device=device,
                        store_batch_size = store_batch_size, query_batch_size = query_batch_size,
                        dim_batch_size = dim_batch_size)
    for i in range(num_groups):
        p = torch.rand((num_points_per_group, dims), dtype=dtype, device=device)
        ids = idx.insert(p)
        found_ids = idx.query_points(p)
        assert sorted(found_ids) == sorted(ids), "RTree index did not return correct ids"
        pass

if __name__ == "__main__":
    # test_storage()    
    # test_rcos_index()
    # test_scor_index()
    # pass
    # test_rtree_index()
    # test_int_index()
    # pass
    test_idx_distr(idxb = partial(RTreeIndex, max_children = 10))
    pass
    # test_time(partial(test_spatial_index_range,
    #                     # index = SpatialIndex,
    #                     # index = partial(GridIndex, epsilons=1, max_bin_size = 256,
    #                     #                 switch_to_all_cap = 0.9),
    #                     # index = partial(RTreeIndex, max_children=256)
    #                   ), 
    #             store_batch_size = [2048, ], #2048, 4096], 
    #             dim_batch_size = [4, ], # 8, 16],
    #             )    
    # test_time(partial(test_spatial_index_query,
    #                     index = SpatialIndex,
    #                     # index = partial(GridIndex, epsilons=1, switch_to_all_cap = 0.9),
    #                     # index = RTreeIndex
    #                   ), 
    #             store_batch_size = [1024], #2048, 4096], 
    #             query_batch_size = [32, ], # 512, 1024],
    #             dim_batch_size = [16, 32, 64, 128, 256, 512, 1024],
    #             # dim_batch_size = [1024],
    #             # dim_batch_size = [128],
    #             # max_children = [16, 32, 64, 128, 256], # 2048, 4096],
    #             )
    
    # test_time2(test_spatial_index_query, 
    #             dict(_name = "SpatialIndex",
    #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256),
    #             dict(_name = "GridIndex",
    #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 128,
    #                     max_children = 64),
    #             # dict(_name = "GridIndex2",
    #             #         store_batch_size = 1024, query_batch_size = 32, dim_batch_size = 64,
    #             #         max_children = 64),                        
    #             dict(_name = "RTreeIndex",
    #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256, 
    #                     max_children = 64),
    #             # dict(_name = "RTreeIndex2",
    #             #         store_batch_size = 1024, query_batch_size = 32, dim_batch_size = 256, 
    #             #         max_children = 64),                        
    #     )

    pass 

    test_time_all(test_spatial_index_query2,
        indices = {
            "default": partial(SpatialIndex, 
                                store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256),
            "int:1": partial(InteractionIndex, target = 0.5,
                                 store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),
            "rcos:1": partial(RCosIndex, target = 0.5, dtype = torch.float32,
                                 store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),                                 
            # "grid:1": partial(GridIndex, 
            #                     store_batch_size = 512, query_batch_size = 128, dim_batch_size = 512, max_children = 256),
            # "grid:2": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 512, max_children = 256),
            # "grid:3": partial(GridIndex, 
            #                     store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 256),
            # "grid:4": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 1024, max_children = 256),
            # "grid:5": partial(GridIndex, 
            #                     store_batch_size = 512, query_batch_size = 128, dim_batch_size = 512, max_children = 64),
            # "grid:6": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 512, max_children = 64),
            "grid:7": partial(GridIndex, 
                                store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),
            # "grid:8": partial(GridIndex, 
            #                     store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 1024, max_children = 64)

            "rtree:1": partial(RTreeIndex, 
                                store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 256, max_children = 64),
            "rtree:2": partial(RTreeIndex, 
                                store_batch_size = 1024, query_batch_size = 128, dim_batch_size = 1024, max_children = 64),

            'spear:1': partial(SpearmanCorIndex, target = 0.5, dtype = torch.float32,
                                 store_batch_size = 512, query_batch_size = 128, dim_batch_size = 1024, max_children = 4000),
        })

    ## NOTE: best Spatial dim_batch_size=256     max_children = *
    ## NOTE: best RTree   dim_batch_size=256    max_children = 64
    ## NOTE: best Grid    dim_batch_size=128     max_children = 128/64
    # test_time(partial(test_spatial_index,
    #                     # index = SpatialIndex,
    #                     # index = partial(GridIndex, epsilons=1, max_bin_size = 64),
    #                     index = partial(RTreeIndex, max_children=35)
    #                   ), 
    #             store_batch_size = [1024, ], #2048, 4096], 
    #             dim_batch_size = [4, ], # 8, 16],
    #             )
    # plt.ioff()
    pass