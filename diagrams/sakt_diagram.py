from graphviz import Digraph

def create_sakt_digraph():
    dot = Digraph(comment='Sakt Diagram')
    dot.attr(rankdir = 'TB', size = '8,5')
    dot.attr('node', shape='oval')