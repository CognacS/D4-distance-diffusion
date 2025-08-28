from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Metric
import wandb
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


from src.evaluation.assignment.core import Assignment, ClonableWithSplitsMixin
from logging import Logger

from copy import copy


class Generator(ABC, pl.LightningModule):

    IGNORED_HPARAMS = [
        'dataset_info',
        'test_assignment',
        'console_logger'
    ]

    def __init__(
            self,
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):
        super().__init__()

        self.dataset_info = dataset_info
        self.test_assignment = test_assignment
        self.console_logger = console_logger
        


    @abstractmethod
    def sample(
        self,
        num_samples: int,
        conditioning_elems: Optional[Dict]=None,
        **kwargs
    ):
        raise NotImplementedError
    



class GeneratorWithEvaluation(Generator):


    def __init__(
            self,
            validation: Dict,
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):

        super().__init__(
            dataset_info=dataset_info,
            test_assignment=test_assignment,
            console_logger=console_logger
        )

        ####################  VALIDATION ASSIGNMENT SETUP  #####################
        self.validation_config = validation

        if self.validation_config['do_assignment']:
            self.add_valid_assignment()
        else:
            self.valid_assignment = None

        ############################  EXTRA SETUP  #############################
        self.start_time = time.time()
        self.total_elapsed_time = 0
        self.max_memory_reserved = 0


    def metrics_to_paths_structure(self, metrics: Dict):
        """
        Convert metrics dictionary to a paths structure
        """
        out_metrics = {}

        for name, m in metrics.items():
            if isinstance(m, dict):
                for k, v in m.items():
                    out_metrics[f'{name}/{k}'] = v
            else:
                out_metrics[name] = m

        return out_metrics
    

    def apply_prefix(self, metrics, prefix):
        """
        Build a paths structure of a dictionary of metrics, and apply a prefix to the paths
        """
        out_metrics = self.metrics_to_paths_structure(metrics)
        return {f'{prefix}/{k}'.lower(): v for k, v in out_metrics.items()}


    def log_wandb_objects(self, objects):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_metrics(objects)
        #wandb.log({name: wb_object})

    
    def log_wandb_histograms(self, histograms):
        if isinstance(self.logger, WandbLogger):
            # wandb accepts a maximum number of 512 bins
            hists = {}
            for k, v in histograms.items():
                if len(v[0]) <= 512:
                    hists[k] = wandb.Histogram(np_histogram=v)
                else:
                    self.console_logger.warning(f'Cannot log histogram {k} because it has more than 512 bins')
            self.log_wandb_objects(hists)


    def get_metrics_values(self, metrics):
        ret = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                ret.update(self.get_metrics_values(v))
            elif isinstance(v, Metric):
                ret[k] = v.compute().detach().cpu().item()
            else:
                ret[k] = v

        return ret
        


    def add_valid_assignment(self):
        if self.test_assignment is None:
            self.valid_assignment = None
            return

        if isinstance(self.test_assignment, ClonableWithSplitsMixin):
            self.console_logger.info('Creating a validation assignment with the same metrics as the test assignment')
            self.valid_assignment = self.test_assignment.clone_with_another_split('valid')
        else:
            self.console_logger.warning('Validation assignment will be the same as the test assignment')
            self.valid_assignment = copy(self.test_assignment)

        self.valid_assignment.how_many_to_generate = self.validation_config['how_many_to_generate']

        


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        overtime = 0 if self.start_time is None else time.time() - self.start_time
        checkpoint['total_elapsed_time'] = self.total_elapsed_time + overtime
        checkpoint['max_memory_reserved'] = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.total_elapsed_time = checkpoint['total_elapsed_time']
        self.max_memory_reserved = checkpoint['max_memory_reserved']
        
    
    def concatenate_conditioning_elems(self, conditioning_elems, num_samples: int):
        pass


    @torch.no_grad()
    def compute_sampling_data(self, num_samples, conditioning_elems=None, sampling_kwargs=None):

        self.console_logger.info('Sampling some graphs...')

        if conditioning_elems is not None:

            conditioning_elems = self.concatenate_conditioning_elems(
                conditioning_elems,
                num_samples=num_samples
            )
        
        # initialize for process metrics
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()
        
        sampling_kwargs = sampling_kwargs if sampling_kwargs is not None else {}

        # sample required graphs
        samples = self.sample(
            num_samples = num_samples,
            conditioning_elems = conditioning_elems,
            **sampling_kwargs
        )

        # end for process metrics
        end_time = time.time()
        peak_memory_usage = float(torch.cuda.max_memory_allocated(0))

        self.console_logger.info(f'Done. Sampling took {end_time - start_time:.2f} seconds\n')

        # compute some statistics on the generated graphs
        num_nodes = [s.num_nodes for s in samples]
        num_edges = [s.num_edges for s in samples]
        indegree = [s.indegree.cpu().tolist() for s in samples]
        
        min_indegree = min([min(indeg) for indeg in indegree])
        max_indegree = max([max(indeg) for indeg in indegree])
        mean_indegree = np.mean([np.mean(indeg) for indeg in indegree])
        num_edges_hist_first = np.histogram(indegree[0], bins=np.arange(min_indegree-0.5, max_indegree+1.5), density=True)
        num_edges_hist = np.histogram(np.concatenate(indegree), bins=np.arange(min_indegree-0.5, max_indegree+1.5), density=True)

        # log statistics
        self.console_logger.info(f'Number of nodes per graph: avg:{np.mean(num_nodes)}, min:{np.min(num_nodes)}, max:{np.max(num_nodes)}')
        self.console_logger.info(f'Number of edges per graph: avg:{np.mean(num_edges)}, min:{np.min(num_edges)}, max:{np.max(num_edges)}')
        self.console_logger.info(f'Indegree per graph: avg:{mean_indegree}, min:{min_indegree}, max:{max_indegree}')

        # compute histogram of number of nodes
        num_nodes_hist = np.histogram(num_nodes, bins=np.arange(min(num_nodes)-0.5, max(num_nodes)+1.5), density=True)

        hists = {
            'num_nodes_hist': num_nodes_hist,
            'num_edges_hist_first': num_edges_hist_first,
            'num_edges_hist': num_edges_hist
        }

        sampling_data = {
            'data': samples,
            'comp_data':{
                'sampling':{
                    'time': {'start': start_time, 'end': end_time},
                    'memory': {'peak': peak_memory_usage}
                }
            }
        }

        return sampling_data, hists


    @torch.no_grad()
    def perform_assignment(self, assignment: Assignment=None, conditioning_elems=None, sampling_kwargs=None, other_metrics=None) -> Dict:

        if assignment is None:
            return {}, None

        ######## compute the sampling metrics ########
        num_samples = assignment.how_many_to_generate

        sampling_data, hists = self.compute_sampling_data(
            num_samples = num_samples,
            conditioning_elems = conditioning_elems,
            sampling_kwargs = sampling_kwargs
        )

        ##############################################

        # log computational metrics
        overtime = 0 if self.start_time is None else time.time() - self.start_time
        
        sampling_data['comp_data']['train'] = {
            'total_time': self.total_elapsed_time + overtime,
            'memory':  max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)
        }

        other_metrics = other_metrics if other_metrics is not None else {}
        
        # perform assignment
        assignment_results = assignment(
            **other_metrics,
            **sampling_data
        )

        return assignment_results, hists


    def log_sampled_graphs(self, samples, how_many=10, method='networkx'):
        if how_many > 0:

            how_many = min(how_many, len(samples))

            if method == 'networkx':

                from torch_geometric.utils import to_networkx

                # transform first log_chain graphs of output_batch to networkx
                graphs_to_log = [to_networkx(samples[i], to_undirected=True) for i in range(how_many)]
                imgs = [graph_to_image(g) for g in graphs_to_log]

            if method == 'scene_graphs':
                graphs_to_log = [graph_to_sg_converter.graph_to_scene_graph(samples[i]).graph for i in range(how_many)]
                imgs = [visualize_sg_3d(g, "training", visualize_alone=False) for g in graphs_to_log]
                figs = [visualize_sg_3d_plotly(g, "training") for g in graphs_to_log]

                grid_fig = grid_plotly_figs(figs, cols=3, shared_legend=True)

                self.logger.experiment.log({
                    'generation/graphs_plotly_grid': grid_fig
                })
            else:
                raise ValueError(f'Unknown method {method}')
            
            # transform to wandb images
            images = [wandb.Image(img) for img in imgs]
            # log them
            self.log_wandb_objects({
                'generation/graphs': images
            })





import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import networkx as nx

def graph_to_image(graph: nx.Graph) -> Image:
    plt.figure(figsize=(5, 5))
    nx.draw(graph, with_labels=True)
    plt.axis('off')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

from src.data.datasets import msd

graph_to_sg_converter = msd.SparseGraphToSceneGraphDecoder()

def visualize_sg_3d(graph, image_name, visualize_alone=False, include_node_ids=True, logger=None):
    nodes_data = graph.nodes(data=True)
    viz_center_offsets = {"ws": np.array([0, 0, 0]), "room": np.array([0, 0, 2]), "wall": np.array([0, 0, 1]),\
                                   "floor": np.array([0, 0, 3]), "building": np.array([0, 0, -2]), "object": np.array([0, 0, 0.5])}

    fig = plt.figure(image_name)
    ax = fig.add_subplot(111, projection='3d')

    def to_3d(arr):
        arr = np.array(arr)
        if arr.shape[-1] == 2:
            arr = np.append(arr, 0)
        return arr

    # For legend
    legend_handles = {}
    for node_data in nodes_data:
        if node_data[1]["viz"]["type"] == "Point":
            markersize = node_data[1].get("markersize", 1.0)
            viz_data = to_3d(node_data[1]["viz"]["center"]) + viz_center_offsets[node_data[1]["type"]]
            node_data[1]["viz"]["center"] = viz_data
            color = _mpl_color_from_feat(node_data[1]["viz"]["feat"])
            marker = node_data[1]["viz"]["feat"][1] if len(node_data[1]["viz"]["feat"]) > 1 else 'o'
            label = node_data[1].get("type", "Point")
            # Only add one handle per label
            if label not in legend_handles:
                h = ax.scatter([], [], [], marker=marker, s=markersize*30, color=color, label=label)
                legend_handles[label] = h
            ax.scatter(viz_data[0], viz_data[1], viz_data[2], marker=marker, s=markersize*30, color=color)
            tag_center = viz_data
        elif node_data[1]["viz"]["type"] == "Line":
            viz_data = np.array(node_data[1]["viz"]["limits"])
            linewidth = node_data[1]["viz"].get("linewidth", 1.5)
            color = _mpl_color_from_feat(node_data[1]["viz"]["feat"])
            label = node_data[1]["viz"].get("type", "Line")
            if viz_data.shape[1] == 2:
                viz_data = np.hstack([viz_data, np.zeros((viz_data.shape[0], 1))])
            # Only add one handle per label
            if label not in legend_handles:
                h, = ax.plot([], [], [], color=color, linewidth=linewidth, label=label)
                legend_handles[label] = h
            ax.plot(viz_data[:,0], viz_data[:,1], viz_data[:,2], color=color, linewidth=linewidth)
            center = to_3d(node_data[1]["center"])
            normal = to_3d(node_data[1].get("normal", [0, 0, 0]))
            norm_line = np.stack([center, center + normal/4])
            ax.plot(norm_line[:,0], norm_line[:,1], norm_line[:,2], color='b', linewidth=linewidth)
            tag_center = center + normal * 0.5 if np.linalg.norm(normal) > 0 else center
        if include_node_ids:
            ax.text(tag_center[0], tag_center[1], tag_center[2], str(node_data[0]), fontsize=10, color='black')
    edges_data = graph.edges(data=True)
    for edge_data in edges_data:
        points = np.array([to_3d(nodes_data[edge_data[0]]["viz"]["center"]), to_3d(nodes_data[edge_data[1]]["viz"]["center"])])
        color = _mpl_color_from_feat(edge_data[2].get("viz_feat", "k"))
        linewidth = edge_data[2].get("linewidth", 1.0)
        alpha = edge_data[2].get("alpha", 0.2)
        label = edge_data[2].get("type", "Edge")
        # Only add one handle per label
        if label not in legend_handles:
            h, = ax.plot([], [], [], color=color, linewidth=linewidth, alpha=alpha, label=label)
            legend_handles[label] = h
        ax.plot(points[:,0], points[:,1], points[:,2], color=color, linewidth=linewidth, alpha=alpha)

    ax.set_box_aspect([1,1,1])
    # Add legend
    ax.legend()
    if visualize_alone:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _mpl_color_from_feat(viz_feat):
    """Helper to extract matplotlib color from viz_feat string like 'go', 'ro', etc."""
    if isinstance(viz_feat, str) and len(viz_feat) > 0:
        color_dict = {
            'g': 'green',
            'r': 'red',
            'b': 'blue',
            'k': 'black',
            'c': 'cyan',
            'm': 'magenta',
            'y': 'yellow',
            'o': 'orange'
        }
        c = viz_feat[0]
        return color_dict.get(c, c)
    return 'k'


import numbers, math
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots   # if you use the grid helper

# ----------------------------------------------------------------- helpers
_MPL_SINGLE_LETTER = {"b":"blue","g":"green","r":"red","c":"cyan",
                      "m":"magenta","y":"yellow","k":"black","w":"white"}
_PLOTLY_3D_SYMBOLS = {"o":"circle","s":"square","d":"diamond",
                      "+":"cross","x":"x"}

def _mpl_to_plotly_symbol(code:str)->str:
    return _PLOTLY_3D_SYMBOLS.get(str(code).lower(),"circle")

def _mpl_to_plotly_color(c:Any,fallback="#1f77b4")->str:
    import numpy as np
    if isinstance(c,str):
        c=c.strip()
        if len(c)==1:
            return _MPL_SINGLE_LETTER.get(c.lower(),fallback)
        if c.startswith("#")or c.startswith(("rgb(","hsl(")):
            return c
        return c
    if isinstance(c,(tuple,list,np.ndarray))and len(c)==3:
        arr=np.asarray(c,dtype=float)
        if arr.max()<=1.0:arr=(arr*255).round()
        r,g,b=map(int,arr);return f"rgb({r},{g},{b})"
    if isinstance(c,numbers.Number):
        v=int(np.clip(c,0,1)*255);return f"rgb({v},{v},{v})"
    return fallback
# ----------------------------------------------------------------- main
def visualize_sg_3d_plotly(
        graph,
        figure_title:str|None=None,
        *,
        include_node_ids:bool=True,
        include_normals:bool=True,
        max_edges_per_trace:int=500,
        viz_center_offsets:Dict[str,np.ndarray]|None=None):
    """
    Interactive Plotly clone of Matplotlib `visualize_sg_3d`.
    """
    if viz_center_offsets is None:
        viz_center_offsets={"ws":np.array([0,0,0]),
                            "room":np.array([0,0,2]),
                            "wall":np.array([0,0,1]),
                            "floor":np.array([0,0,3]),
                            "building":np.array([0,0,-2]),
                            "object":np.array([0,0,0.5]),}

    def to_3d(arr):
        arr=np.asarray(arr,dtype=float)
        if arr.shape[-1]==2:
            arr=np.concatenate([arr,[0.0]])
        return arr

    traces=[]
    legend_seen=set()

    # ----------------------------- nodes
    for nid,data in graph.nodes(data=True):
        viz=data["viz"]
        ntype=data.get("type","Point")
        color=_mpl_to_plotly_color(viz["feat"][0])

        if viz["type"]=="Point":
            center=to_3d(viz["center"])+viz_center_offsets.get(ntype,0)
            marker_code=viz["feat"][1] if len(viz["feat"])>1 else "o"
            marker_sym=_mpl_to_plotly_symbol(marker_code)
            size=data.get("markersize",1.0)*4

            tr=go.Scatter3d(
                x=[center[0]],y=[center[1]],z=[center[2]],
                mode="markers+text" if include_node_ids else "markers",
                marker=dict(size=size,color=color,symbol=marker_sym),
                name=ntype,showlegend=False,
                text=[str(nid)] if include_node_ids else None,
                textposition="top center",
                hovertemplate=f"id: {nid}<br>type: {ntype}")
            traces.append(tr)

        elif viz["type"]=="Line":
            # limits polyline
            limits=np.asarray(viz["limits"],dtype=float)
            if limits.shape[1]==2:
                limits=np.column_stack([limits,np.zeros(limits.shape[0])])
            linewidth=viz.get("linewidth",1.5)

            tr=go.Scatter3d(
                x=limits[:,0],y=limits[:,1],z=limits[:,2],
                mode="lines",
                line=dict(color=color,width=linewidth),
                name=viz.get("type","Line"),showlegend=False,
                hoverinfo="skip")
            traces.append(tr)

            # optional normal vector
            if include_normals:
                center=to_3d(data["center"])
                normal=to_3d(data.get("normal",[0,0,0]))
                if np.linalg.norm(normal)>0:
                    p0,p1=center,center+normal/4
                    traces.append(
                        go.Scatter3d(x=[p0[0],p1[0]],y=[p0[1],p1[1]],
                                     z=[p0[2],p1[2]],mode="lines",
                                     line=dict(color="blue",width=linewidth),
                                     name="normal",showlegend=False,
                                     hoverinfo="skip"))
            # tag centre for ID label
            if include_node_ids:
                tag=center+normal*0.5 if np.linalg.norm(normal)>0 else center
                traces.append(
                    go.Scatter3d(
                        x=[tag[0]],y=[tag[1]],z=[tag[2]],
                        mode="text",
                        text=[str(nid)],
                        textposition="top center",
                        hoverinfo="skip",
                        showlegend=False))

        # record first appearance for legend
        if ntype not in legend_seen:
            traces[-1].showlegend=True
            legend_seen.add(ntype)

    # ----------------------------- edges
    xs,ys,zs=[],[],[]
    edge_cnt=0
    for u,v,edata in graph.edges(data=True):
        p0=to_3d(graph.nodes[u]["viz"]["center"])+viz_center_offsets.get(graph.nodes[u]["type"],0)
        p1=to_3d(graph.nodes[v]["viz"]["center"])+viz_center_offsets.get(graph.nodes[v]["type"],0)

        xs+= [p0[0],p1[0],None]
        ys+= [p0[1],p1[1],None]
        zs+= [p0[2],p1[2],None]
        edge_cnt+=1

        if edge_cnt%max_edges_per_trace==0:
            color=_mpl_to_plotly_color(edata.get("viz_feat","k"))
            traces.append(
                go.Scatter3d(x=xs,y=ys,z=zs,mode="lines",
                             line=dict(color=color,
                                       width=edata.get("linewidth",1)),
                             opacity=edata.get("alpha",0.2),
                             name=edata.get("type","Edge"),
                             hoverinfo="skip"))
            xs,ys,zs=[],[],[]

    if xs:
        color=_mpl_to_plotly_color("k")
        traces.append(go.Scatter3d(
            x=xs,y=ys,z=zs,mode="lines",
            line=dict(color=color,width=1),
            opacity=0.2,name="Edge",hoverinfo="skip"))

    # ----------------------------- layout
    fig=go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=figure_title) if figure_title else None,
        scene=dict(aspectmode="data",
                   xaxis_title="",yaxis_title="",zaxis_title=""),
        legend_title_text="",
        margin=dict(l=0,r=0,t=40 if figure_title else 0,b=0))
    return fig



def grid_plotly_figs(figs, *, cols=3, subplot_titles=None, size_per_cell=400,
                     shared_legend=True, **layout_kwargs):
    """
    Stack multiple 3-D Plotly figures into an R×C grid and keep them aligned.
    """
    n = len(figs)
    rows = math.ceil(n / cols)

    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    titles = (subplot_titles or
              [f.layout.title.text or f"#{i}" for i, f in enumerate(figs)])

    grid = make_subplots(rows=rows, cols=cols, specs=specs,
                         horizontal_spacing=0.04, vertical_spacing=0.04,
                         subplot_titles=titles)

    for i, f in enumerate(figs):
        r, c = divmod(i, cols)

        # --- add traces ----------------------------------------------------
        for tr in f.data:
            grid.add_trace(tr, row=r + 1, col=c + 1)

        # --- copy camera & aspect, NOT domain -----------------------------
        scene_id = f"scene{'' if (i == 0) else i + 1}"
        src_scene = f.layout.scene
        dst_scene = grid.layout[scene_id]

        # camera
        if src_scene.camera is not None:
            dst_scene.camera = src_scene.camera

        # aspectmode / aspectratio
        if src_scene.aspectmode is not None:
            dst_scene.aspectmode = src_scene.aspectmode
        if src_scene.aspectratio is not None:
            dst_scene.aspectratio = src_scene.aspectratio

        # you can copy other *non-domain* attributes the same way if desired
        # e.g. dst_scene.xaxis.visible = src_scene.xaxis.visible

    # -------- optional shared legend --------------------------------------
    if shared_legend:
        for tr in grid.data:
            tr.showlegend = False
        seen = set()
        for tr in grid.data:
            if tr.name and tr.name not in seen:
                tr.showlegend = True
                seen.add(tr.name)

    grid.update_layout(
        width=size_per_cell * cols,
        height=size_per_cell * rows + 40,
        margin=dict(l=0, r=0, t=40, b=0),
        **layout_kwargs,
    )
    return grid
