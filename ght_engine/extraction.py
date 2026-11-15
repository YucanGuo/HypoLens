import pandas as pd
import time
from collections import defaultdict

import networkx as nx

from .utils import HypothesisTesting


def new_graph_hypo_result(args, new_graph, result_list, time_used_list):
    time_rating_extraction_start = time.time()

    dataset_functions = {
        "movielens": {"edge": getEdges, "node": getNodes, "path": getPaths},
        "citation": {"edge": getEdges, "node": getNodes, "path": getPaths},
        "yelp": {"edge": getEdges, "node": getNodes, "path": getPaths},
    }

    if args.dataset in dataset_functions:
        dataset_info = dataset_functions[args.dataset]

        if args.hypo == 1:
            total_result = dataset_info["edge"](args, new_graph)
        elif args.hypo == 2:
            total_result = dataset_info["node"](
                args, new_graph, dimension=args.dimension
            )
        elif args.hypo == 3:
            args.total_valid = 0
            args.total_minus_reverse = 0
            total_result = dataset_info["path"](args, new_graph)

        else:
            raise Exception(
                f"Sorry, we don't support hypothesis {args.hypo} for {args.dataset}."
            )
    else:
        raise Exception(f"Sorry, we don't support the dataset {args.dataset}.")

    if str(list(args.attribute.keys())[0]) not in total_result:
        total_result[list(args.attribute.keys())[0]] = []

    result = {}

    import numpy as np
    for attribute, v in total_result.items():
        # NOTE: for hypo == 3 (path), total_result also contains auxiliary
        # "<condition_name>+user_coverage" / "+movie_coverage" entries (see
        # getPaths()) that are NOT part of the tested attribute distribution.
        # valid_edges/sample_means must only reflect the real condition's
        # value list, otherwise they'd be polluted by unrelated coverage
        # counts (very different scale/meaning) averaged together.
        is_real_condition = len(attribute.split("+")) == 1
        if is_real_condition:
            args.valid_edges.append(len(v))
            if hasattr(args, "sample_means") and len(v) != 0:
                args.sample_means.append(float(np.mean(v)))
        if len(v) <= 20:
            args.logger.info(v)
        args.logger.info(f"The sampled graph has {len(v)} relevant patterns.")
        if is_real_condition:
            if args.agg == "mean":
                if len(v) != 0:
                    hypo_result = HypothesisTesting(args, v, 1)
                    result[attribute] = hypo_result
                else:
                    result[attribute] = None

            else:
                args.logger.error(f"Sorry, we don't support {args.agg}.")
                raise Exception(f"Sorry, we don't support {args.agg}.")
        else:
            # pass
            result[attribute] = v
    time_used_list["sample_graph_by_condition"].append(
        round(time.time() - time_rating_extraction_start, 2)
    )

    if args.hypo == 3:
        result["total_valid"] = args.total_valid
        result["total_minus_reverse"] = args.total_minus_reverse
        now = time.time()
        args.logger.info("start finding diameter and density")
        print("start finding diameter and density")
        connected = nx.is_connected(new_graph)
        if connected:
            result["diameter"] = 10  # nx.diameter(new_graph)
        else:
            result["diameter"] = float("nan")
        args.logger.info(
            f"computing diameter {result['diameter']} takes {round(time.time() - now, 2)}"
        )
        print(
            f"computing diameter {result['diameter']} takes {round(time.time() - now, 2)}"
        )
        result["density"] = nx.density(new_graph)
    result_list.append(result)
    return result_list


def getNodes(args, new_graph, dimension):  # dimension = {"movie":"genre"}
    total_result = defaultdict(list)
    total_result_repeat = defaultdict(list)
    if len(list(args.attribute)) == 1:
        for condition_name, condition_dict in args.attribute.items():
            # The condition is on the dimension node only!
            if len(condition_dict) == 3 and list(dimension.keys())[0] in condition_dict:
                attribute_condition_dict = condition_dict[list(dimension.keys())[0]]
                node_attribute = nx.get_node_attributes(new_graph, "label")

                # for every node, check conditions
                for key, value in node_attribute.items():
                    flag = True
                    if new_graph.nodes[key]["label"] == list(dimension.keys())[0]:
                        for k, v in attribute_condition_dict.items():
                            # str-cast both sides: numeric filters (e.g. year:1990)
                            # must match the same value stored as a string on the graph.
                            if k in new_graph.nodes[key] and str(new_graph.nodes[key][k]) == str(v):
                                pass
                            else:
                                flag = False
                    else:
                        flag = False
                    if flag:
                        total_result[condition_name].append(
                            new_graph.nodes[key][list(dimension.values())[0]]
                        )
            # The condition is on the edge or the other nodes
            # so we need to extract edges to filter
            else:
                selected_edge = condition_dict["edge"]
                edge_dict = nx.get_edge_attributes(new_graph, name=selected_edge)

                for key, value in edge_dict.items():
                    from_node, to_node = key
                    flag = checkCondition(args, condition_dict, new_graph, from_node)
                    if flag:
                        flag = checkCondition(args, condition_dict, new_graph, to_node)
                    else:
                        continue

                    if flag:
                        if (
                            new_graph.nodes[from_node]["label"]
                            == list(dimension.keys())[0]
                        ):  # dimension = {"movie":"genre"}
                            total_result_repeat[condition_name].append(from_node)
                        elif (
                            new_graph.nodes[to_node]["label"]
                            == list(dimension.keys())[0]
                        ):
                            total_result_repeat[condition_name].append(to_node)
                        else:
                            raise Exception("Wrong code")
                distinct_nodes = list(set(total_result_repeat[condition_name]))
                for i in distinct_nodes:
                    assert new_graph.nodes[i]["label"] == list(dimension.keys())[0]
                    total_result[condition_name].append(
                        new_graph.nodes[i][list(dimension.values())[0]]
                    )

    else:
        raise Exception("Sorry we only support one condition_name")
    return total_result


def checkCondition(args, condition_dict, new_graph, node_index):
    if new_graph.nodes[node_index]["label"] in condition_dict:
        attribute_condition_dict = condition_dict[new_graph.nodes[node_index]["label"]]
        for k, v in attribute_condition_dict.items():  # {"gender":"M"}}
            # str-cast both sides: numeric filters (e.g. year:1990) must
            # match the same value stored as a string on the graph.
            if k in new_graph.nodes[node_index] and str(new_graph.nodes[node_index][k]) == str(v):
                pass
            else:
                return False
    else:
        return True
    return True


def getEdges(args, new_graph):
    # Single condition_name supported (matches getPaths).
    if len(list(args.attribute)) != 1:
        raise Exception("Sorry we only support one condition_name")
    condition_name, condition_dict = next(iter(args.attribute.items()))

    total_result = defaultdict(list)

    # ── Node mode: TEST target is a node attribute of one endpoint.
    # condition_dict = {"node": {"index": 0|1, "attribute": attr},
    #                   "path": [{type, attribute}, {type, attribute}],
    #                   <label>: {filters}}
    # Mirror getPaths node mode: orient each graph edge to the pattern's
    # traversal order (pos 0 → path_info[0]["type"], pos 1 → the other),
    # check both endpoint conditions, then extract the chosen endpoint's
    # attribute. Handles both directed-stored and reverse-stored edges, and
    # the same-label case (falls back to tuple order).
    if "node" in condition_dict:
        node_spec    = condition_dict["node"]
        pos          = node_spec["index"]          # 0 = first node in chain, 1 = second
        extract_attr = node_spec["attribute"]
        path_info    = condition_dict["path"]
        lbl_0        = path_info[0]["type"]
        lbl_1        = path_info[1]["type"]

        for u, v in new_graph.edges():
            lu = new_graph.nodes[u]["label"]
            lv = new_graph.nodes[v]["label"]
            if lu == lbl_0 and lv == lbl_1:
                a, b = u, v
            elif lu == lbl_1 and lv == lbl_0:
                a, b = v, u                  # reverse orientation
            else:
                continue                      # edge doesn't connect the two pattern types
            chosen = a if pos == 0 else b
            if not checkCondition(args, condition_dict, new_graph, a):
                continue
            if not checkCondition(args, condition_dict, new_graph, b):
                continue
            if extract_attr in new_graph.nodes[chosen]:
                total_result[condition_name].append(
                    new_graph.nodes[chosen][extract_attr]
                )
        return total_result

    # ── Edge mode (original): TEST target is an edge attribute.
    # condition_dict["edge"] names the edge attribute to extract.
    selected_edge = condition_dict["edge"]
    edge_dict = nx.get_edge_attributes(new_graph, name=selected_edge)
    for key, value in edge_dict.items():
        from_node, to_node = key
        if not checkCondition(args, condition_dict, new_graph, from_node):
            continue
        if not checkCondition(args, condition_dict, new_graph, to_node):
            continue
        total_result[condition_name].append(value)
    return total_result


def find_paths(
    args, graph, path_info, current_path, current_node, depth, path_count_map
):
    if depth == len(path_info):
        str_current_path = [str(i) for i in current_path]
        key = ".".join(str_current_path)
        str_current_path.reverse()
        reverse_key = ".".join(str_current_path)
        args.total_valid += 1
        if path_count_map.get(key) is None and path_count_map.get(reverse_key) is None:
            args.total_minus_reverse += 1
            if len(set(current_path)) == len(current_path):
                path_count_map[key] = 1
        return

    current_label = path_info[depth]["type"]
    current_conditions = path_info[depth]["attribute"]

    for neighbor in graph.neighbors(current_node):
        neighbor_label = graph.nodes[neighbor]["label"]

        if neighbor_label == current_label:
            # Check conditions for the current node
            flag = True
            # for citation 3-1-1 vague match
            for k, v in current_conditions.items():
                if pd.isna(pd.isna(graph.nodes[neighbor][k])):
                    flag = False
                    break
                # Cast both sides to string: handles numeric attrs stored as
                # strings on one side, and preserves the citation 3-1-1 vague
                # substring match.
                elif (str(graph.nodes[neighbor][k]) != str(v)
                      and str(v) not in str(graph.nodes[neighbor][k])):
                    flag = False
                    break

            if flag:
                # Recursively explore the next depth
                find_paths(
                    args,
                    graph,
                    path_info,
                    current_path + [neighbor],
                    neighbor,
                    depth + 1,
                    path_count_map,
                )


def getPaths(args, new_graph):
    total_result = defaultdict(list)
    # path_count_map is a hash map: keys = path node index join by "." and values = 1
    path_count_map = {}
    if len(list(args.attribute)) == 1:
        for condition_name, condition_dict in args.attribute.items():
            assert (
                "path" in condition_dict
            ), f"hypo {args.hypo} must require path information in args.attribute."
            path_info = condition_dict["path"]

            for ini_node in new_graph.nodes():
                if new_graph.nodes[ini_node]["label"] == path_info[0]["type"]:
                    flag = True
                    for k, v in path_info[0]["attribute"].items():
                        # for citation 3-1-1 vague match
                        if pd.isna(new_graph.nodes[ini_node][k]):
                            flag = False
                            break
                        # Cast both sides to string: handles numeric attrs
                        # stored as strings on one side, and preserves the
                        # citation 3-1-1 vague substring match.
                        elif (str(new_graph.nodes[ini_node][k]) != str(v)
                              and str(v) not in str(new_graph.nodes[ini_node][k])):
                            flag = False
                            break
                    if flag:
                        find_paths(
                            args,
                            new_graph,
                            path_info,
                            [ini_node],
                            ini_node,
                            1,
                            path_count_map,
                        )
            args.logger.info("finished path extraction!")
            print("finished path extraction!")

            user_set = set()
            movie_set = set()
            if "edge_diff" in condition_dict.keys():
                # edge_diff: signed difference of the same attribute on two
                # positional edges of each matched path (paired comparison;
                # no abs).  indices are edge positions along the chain:
                # edge i connects r[i] — r[i+1].
                edge_spec = condition_dict["edge_diff"]
                extract_edge_attr = edge_spec["attribute"]
                pos_i, pos_j = edge_spec["indices"]
                for r in path_count_map.keys():
                    r = list(map(int, r.split(".")))
                    r = [int(i) for i in r]
                    # Coverage: all intermediate nodes are "user-like",
                    # all endpoint nodes are "movie-like".
                    user_set.update(r[1:-1])
                    movie_set.update([r[0], r[-1]])
                    e_i = (r[pos_i], r[pos_i + 1])
                    e_j = (r[pos_j], r[pos_j + 1])
                    if (extract_edge_attr in new_graph.edges[e_i].keys()
                            and extract_edge_attr in new_graph.edges[e_j].keys()):
                        total_result[condition_name].append(
                            new_graph.edges[e_i][extract_edge_attr]
                            - new_graph.edges[e_j][extract_edge_attr]
                        )
            elif "edge" in condition_dict.keys():
                extract_edge_attr = condition_dict["edge"]
                for r in path_count_map.keys():
                    r = list(map(int, r.split(".")))
                    r = [int(i) for i in r]
                    # Coverage: all intermediate nodes are "user-like",
                    # all endpoint nodes are "movie-like".
                    user_set.update(r[1:-1])
                    movie_set.update([r[0], r[-1]])
                    res = []
                    for index in range(len(r) - 1):
                        e = (r[index], r[index + 1])
                        if extract_edge_attr in new_graph.edges[e].keys():
                            res.append(new_graph.edges[e][extract_edge_attr])

                    # average
                    total_result[condition_name].append(sum(res) / len(res))
            elif (
                "node" in condition_dict.keys()
            ):  # node in dict, "node": {"index":pos,"attribute":"age"}
                node_spec = condition_dict["node"]
                extract_node_attr = node_spec["attribute"]
                mode = condition_dict.get("mode", "mean")

                for r in path_count_map.keys():
                    r = list(map(int, r.split(".")))
                    r = [int(i) for i in r]
                    user_set.update(r[1:-1])
                    movie_set.update([r[0], r[-1]])

                    if mode == "diff":
                        # node_diff: positional indices specify the two nodes
                        pos_a, pos_b = node_spec["indices"]
                        va = new_graph.nodes[r[pos_a]][extract_node_attr]
                        vb = new_graph.nodes[r[pos_b]][extract_node_attr]
                        total_result[condition_name].append(abs(va - vb))
                    else:
                        # target.node: positional index specifies the node
                        pos = node_spec["index"]
                        total_result[condition_name].append(
                            new_graph.nodes[r[pos]][extract_node_attr]
                        )

            else:
                raise Exception("You must provide the dimension in the attribute.")
            total_result[condition_name + "+user_coverage"].append(len(user_set))
            total_result[condition_name + "+movie_coverage"].append(len(movie_set))

    else:
        raise Exception("Sorry we only support one condition_name")
    return total_result
