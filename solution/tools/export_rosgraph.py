#!/usr/bin/env python3
"""Export the live ROS graph to rosgraph.png using rqt_graph's dotcode
generator (the same code path as the rqt_graph GUI's save-as-image)."""
import subprocess
import sys
import time

import rclpy

from rqt_graph.dotcode import RosGraphDotcodeGenerator
from rqt_graph import rosgraph2_impl
from qt_dotgraph.pydotfactory import PydotFactory


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'rosgraph.png'
    rclpy.init()
    node = rclpy.create_node('rosgraph_exporter')

    # Let the node discover the graph (DDS discovery with many nodes takes a
    # while), then update the graph model a few times.
    graph = rosgraph2_impl.Graph(node)
    graph.set_node_stale(30.0)
    for _ in range(4):
        end = time.time() + 4.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        graph.update()

    gen = RosGraphDotcodeGenerator(node)
    dotcode = gen.generate_dotcode(
        rosgraphinst=graph,
        ns_filter='/',
        topic_filter='/',
        graph_mode='node_topic',
        dotcode_factory=PydotFactory(),
        hide_single_connection_topics=False,
        hide_dead_end_topics=True,
        cluster_namespaces_level=2,
        accumulate_actions=True,
        orientation='LR',
        quiet=True,
        hide_tf_nodes=True,
    )

    dot_path = '/tmp/rosgraph.dot'
    with open(dot_path, 'w') as f:
        f.write(dotcode)
    subprocess.run(['dot', '-Tpng', dot_path, '-o', out], check=True)
    print(f'wrote {out}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
