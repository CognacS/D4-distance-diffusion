##############################################################################################
#  INSPIRED FROM https://github.com/cvignac/DiGress/blob/main/src/analysis/visualization.py  #
##############################################################################################

from typing import List, Union, Dict, Tuple

import os

import numpy as np
from torch import Tensor
import networkx as nx

from PIL import Image as PIL
try:
    from IPython.display import display
    _IPYTHON_INSTALLED = True
except ImportError:
    _IPYTHON_INSTALLED = False

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import io
import imageio

from rdkit.Chem import Draw

import wandb

from src.datatypes.sparse import SparseGraph, to_directed

from src.data.simple_transforms.molecular import GraphToMoleculeConverter
from torch_geometric.utils.convert import to_networkx

def _save_image(path, name, img: PIL.Image):
    img.save(os.path.join(path, f'{name}.png'))

def _save_video(path, name, video: List[PIL.Image]):
    imageio.mimsave(os.path.join(path, f'{name}.mp4'), video, fps=5)


def _log_image(name, img: PIL.Image, wandb_image_kwargs: Dict):
    wandb.log({f'graph/{name}': wandb.Image(img, **wandb_image_kwargs)}, commit=False)

def _log_video(name, video: List[PIL.Image], wandb_video_kwargs: Dict):
    wandb.log({f'chain/{name}': wandb.Video(video, **wandb_video_kwargs)}, commit=False)


def _show_image(img: PIL.Image):
    if _IPYTHON_INSTALLED:
        display(img)

def _show_video(video: List[PIL.Image]):
    fig = plt.figure()
    frames = []
    for frame in video:
        img = plt.imshow(np.array(frame), animated=True)
        frames.append([img])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                repeat_delay=1000)
    plt.show()


PRINTERS = {
    'save': {
        'image': _save_image,
        'video': _save_video
    },
    'wandb': {
        'image': _log_image,
        'video': _log_video
    },
    'console': {
        'image': _show_image,
        'video': _show_video
    }
}


class BaseVisualizer:

    def __init__(self, name: str):
        self.name = name

    def as_image(self, data: SparseGraph) -> Union[PIL.Image, np.ndarray]:
        raise NotImplementedError



class VisualizationPrinter:

    def __init__(
            self,
            visualizers: List[BaseVisualizer],
            # printers flags
            store_on_disk: bool = False,
            log_to_wandb: bool = False,
            display_on_console: bool = False,
            # printers kwargs
            root_path: str = None,
            wandb_image_kwargs: Dict = None,
            wandb_video_kwargs: Dict = None
        ):
        self.visualizers = visualizers
        
        self.store_on_disk = store_on_disk
        self.log_to_wandb = log_to_wandb
        self.display_on_console = display_on_console

        if self.store_on_disk:
            assert root_path is not None, 'root_path must be provided if store_on_disk is True'
        self.root_path = root_path
        self.wandb_kwargs = {}
        self.wandb_kwargs['image'] = wandb_image_kwargs if wandb_image_kwargs is not None else {}
        self.wandb_kwargs['video'] = wandb_video_kwargs if wandb_video_kwargs is not None else {'fps':5, 'format': "gif"}


    def print(
            self,
            data: Union[SparseGraph, List[SparseGraph], List[List[SparseGraph]]],
            suffix: str='',
            only_video: bool = False
        ):

        # now data can be either List[SparseGraph] or List[List[SparseGraph]]
        data = unbatch_graphs(data)

        # if data is a list of graphs, visualize each element as a separate image
        if isinstance(data[0], SparseGraph):
            for i, graph in enumerate(data):
                self._visualize_graph(graph, suffix=f'{suffix}_{i}')
                
        # if data is a list of list of graphs, visualize each element as a separate video
        # where each list is the evolution of a graph
        elif isinstance(data[0], list):
            for i, graphs in enumerate(data):
                # prepare lists for videos
                imgs = {vis.name: [] for vis in self.visualizers}
                for j, graph in enumerate(graphs):
                    graph = graph.clone().collapse()
                    # gather images for each visualizer
                    curr_imgs = self._visualize_graph(
                        graph,
                        suffix=f'{suffix}_{i}_{j}',
                        return_images = True,
                        call_printers = not only_video
                    )
                    for vis_name, img in curr_imgs.items():
                        imgs[vis_name].append(img)
                # create videos
                for vis_name, vis_imgs in imgs.items():
                    self._call_printers(f'{vis_name}_{suffix}_{i}', vis_imgs, 'video')

        if self.log_to_wandb:
            wandb.log({})

        

    def _visualize_graph(
            self,
            graph: SparseGraph,
            suffix: str='',
            return_images: bool = False,
            call_printers: bool = True
        ):
        imgs = {}

        for visualizer in self.visualizers:

            # visualize image
            img = visualizer.as_image(graph)

            # give the image a name
            name = f'{visualizer.name}'
            if suffix != '': name += f'_{suffix}'

            if call_printers:
                # call printers (save, log, display)
                self._call_printers(name, img, 'image')

            # store image if needed (e.g. for video creation)
            if return_images:
                imgs[visualizer.name] = img

        if return_images:
            return imgs
    

    def _call_printers(self, name, media, media_type):
        if self.store_on_disk:
            PRINTERS['save'][media_type](self.root_path, name, media)
        if self.log_to_wandb:
            PRINTERS['wandb'][media_type](name, media, self.wandb_kwargs[media_type])
        if self.display_on_console:
            PRINTERS['console'][media_type](media)



class MoleculeVisualizer(BaseVisualizer):

    def __init__(self, name: str, atoms_decoder: Dict, bonds_decoder: Dict):
        super().__init__(name)

        self.graph_to_mol = GraphToMoleculeConverter(atoms_decoder, bonds_decoder)


    def as_image(self, data: SparseGraph) -> PIL.Image:
        mol = self.graph_to_mol(data)
        return Draw.MolToImage(mol, size=(300, 300))


class GraphVisualizer(BaseVisualizer):

    def __init__(
            self, name: str,
            nodes_decoder: Dict = None,
            edges_decoder: Dict = None,
            nodes_cmap: str = None,
            edges_cmap: str = None,
            undirected: bool = False
        ):
        super().__init__(name)

        self.nodes_decoder = nodes_decoder
        self.edges_decoder = edges_decoder

        self.nodes_cmap = plt.get_cmap(nodes_cmap, len(self.nodes_decoder)) if nodes_cmap is not None else None
        self.edges_cmap = plt.get_cmap(edges_cmap, len(self.edges_decoder)) if edges_cmap is not None else None

        self.undirected = undirected


    def as_image(self, data: SparseGraph) -> PIL.Image:

        if self.undirected and data.is_undirected():
            data = data.clone()
            data.edge_index, data.edge_attr = to_directed(data.edge_index, data.edge_attr)

        # convert graph to networkx
        gx = to_networkx(data, to_undirected=self.undirected)

        # create a matplotlib figure
        fig = plt.figure()

        #############################  DRAW GRAPH  #############################
        pos = nx.kamada_kawai_layout(gx)
        opt_col_names = ['node_color', 'edge_color']
        opt_col_funcs = [self.nodes_cmap, self.edges_cmap]
        opt_col_values = [data.x, data.edge_attr]

        options = {
            **{
                opt_name: opt_func(opt_val)
                for opt_name, opt_func, opt_val in zip(
                    opt_col_names, opt_col_funcs, opt_col_values
                ) if opt_func is not None
            }
        }
        # draw figure
        nx.draw(
            gx, pos,
            with_labels=False,
            **options
        )
        if self.nodes_decoder is not None:
            # draw node labels
            nx.draw_networkx_labels(
                gx, pos,
                labels = nodes_map_fn(data.x, self.nodes_decoder)
            )
        if self.edges_decoder is not None:
            # draw edge labels
            nx.draw_networkx_edge_labels(
                gx, pos,
                edge_labels = edges_map_fn(data.edge_index, data.edge_attr, self.edges_decoder)
            )

        #####################  CONVERT FIGURE TO PIL IMAGE  ####################
        # Convert the figure to a PIL image
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='PNG')
        image_stream.seek(0)
        pil_image = PIL.open(image_stream)

        pil_image.info['dpi'] = str(pil_image.info['dpi'])

        # Close the figure to release the memory
        plt.close(fig)

        return pil_image


def nodes_map_fn(x: Tensor, nodes_decoder: Dict) -> Dict:
    return {node: nodes_decoder[attr.item()] for node, attr in enumerate(x)}

def edges_map_fn(edge_index: Tensor, edge_attr: Tensor, edges_decoder: Dict) -> Dict:
    return {(edge[0].item(), edge[1].item()): edges_decoder[attr.item()] for edge, attr in zip(edge_index.T, edge_attr)}

def get_colors_map(data: Tensor, cmap) -> np.ndarray:
    return cmap(data)
    

def unbatch_graphs(data: SparseGraph) -> List[SparseGraph]:
    if isinstance(data, SparseGraph):
        if hasattr(data, 'ptr'):
            data = data.to_data_list()
        else:
            data = [data]

    return data