import dearpygui.dearpygui as dpg

from demo.demo_mnist import MNISTDemoTab
from demo.demo_faces import FacesDemoTab
from demo.demo_manifold_1d import Manifold1DTab
from demo.demo_synthetic import SyntheticDemoTab
from demo.demo_tda import TopologyDemoTab

class MainApplication:
    def __init__(self):
        dpg.create_context()

        self.synthetic_tab = SyntheticDemoTab()
        self.manifold_1d_tab = Manifold1DTab()
        self.mnist_tab = MNISTDemoTab()
        self.faces_tab = FacesDemoTab()
        self.topology_tab = TopologyDemoTab()

    def build_ui(self):
        with dpg.window(tag="PrimaryWindow"):
            dpg.add_text("Learning Data Densities and Data Coordinate Systems for Disentangled Representations", color=[0, 255, 255])
            dpg.add_spacer(height=5)

            with dpg.tab_bar(tag="MainTabBar"):

                with dpg.tab(label="Synthetic Data"):
                    self.synthetic_tab.build_tab_ui()
                with dpg.tab(label="Loop Detection"):
                    self.manifold_1d_tab.build_tab_ui()
                with dpg.tab(label="MNIST Rotation"):
                    self.mnist_tab.build_tab_ui()
                with dpg.tab(label="Faces"):
                    self.faces_tab.build_tab_ui()
                with dpg.tab(label="Topology (TDA)"):
                    self.topology_tab.build_tab_ui()

        self.mnist_tab.start_workers()
        self.faces_tab.start_workers()
        dpg.set_primary_window("PrimaryWindow", True)

    def run(self):
        self.build_ui()

        dpg.setup_dearpygui()
        dpg.create_viewport(title="Manifold Exploration Suite", width=1250, height=750)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

