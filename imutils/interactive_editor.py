import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage.segmentation import find_boundaries
from skimage.draw import polygon as draw_polygon

class InteractiveEditor:
    def __init__(self, event_callback):
        self.callback = event_callback
        self.base_img = None
        self.img_shape = None
        
        # --- Mode and Data Stores ---
        self.mode = 'mask'
        self.point_labels = []
        self.masks = np.zeros((1, 1), dtype=np.uint16)
        self.object_labels = {}
        
        # --- Point Mode Specific ---
        self.POINT_LABEL_MAP = {1: 'alive', 2: 'dead'}
        self.POINT_COLOR_MAP = {'alive': 'lime', 'dead': 'red'} # Brighter colors for single channel views
        
        # --- State Variables ---
        self.predictions = {}
        self.in_merge_mode = False
        self.merge_candidates = []
        self.in_draw_mode = False
        self.is_actively_drawing = False
        self.draw_points = []
        self.mask_display_mode = 'rings'
        self.color_cache = {}
        self.in_label_mode = False
        self.label_view_mode = 'off'
        self.LABEL_MAP = {0: 'unlabelled', 1: 'alive', 2: 'dead', 3: 'junk'}
        self.COLOR_MAP = {'unlabelled': [0, 0, 1], 'alive': [0, 1, 0], 'dead': [1, 0, 0], 'junk': [0.5, 0.5, 0.5]}
        
        self.draw_artist = None
        self.background = None
        self.active_ax = None # Track which axis is being drawn on
        self.help_text_artist = None
        
        # --- Matplotlib Setup (2x2 Grid) ---
        # sharex/sharey ensures zooming one zooms ALL
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
        self.ax_flat = self.axes.flatten() # 0: Composite, 1: Phase, 2: Alive, 3: Dead
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda evt: self.callback('exit'))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        
        # Set titles once
        titles = ["Composite (RGB)", "Phase (Structure)", "Alive (Green Ch)", "Dead (Red Ch)"]
        for ax, t in zip(self.ax_flat, titles):
            ax.set_title(t, fontsize=10, fontweight='bold')
            ax.axis('off') # Hide axes ticks for cleaner look

        plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.01, right=0.99, top=0.95, bottom=0.01)

    def update_data(self, image: np.ndarray, title: str, mode: str = 'mask', masks: np.ndarray = None, labels: dict = None, point_labels: list = None):
        self.mode = mode
        self.base_img = image # Expecting (H, W, 3) Float32
        self.img_shape = image.shape
        
        # Reset state
        self.predictions = {}
        self.in_merge_mode = False
        self.in_draw_mode = False

        if self.mode == 'point':
            self.point_labels = point_labels.copy() if point_labels else []
            self.masks = np.zeros((1, 1), dtype=np.uint16)
            self.object_labels = {}
        else: # 'mask' mode
            self.point_labels = []
            self.masks = masks.copy() if masks is not None else np.zeros(self.img_shape[:2], dtype=np.uint16)
            self.object_labels = labels.copy() if labels else {int(i): 0 for i in np.unique(self.masks) if i != 0}
            self.label_view_mode = 'labels' if self.in_label_mode else 'off'

        # Set the main window title
        self.fig.canvas.manager.set_window_title(title)
        self._redraw_canvas(maintain_zoom=False)

    def get_masks(self) -> np.ndarray: return self.masks.copy()
    def get_annotations(self) -> dict: return self.object_labels.copy()
    def get_point_labels(self) -> list: return self.point_labels.copy()

    def display_predictions(self, predictions: dict):
        if self.mode != 'mask': return
        self.predictions = predictions
        if self.label_view_mode == 'predictions':
            print(f"Background predictions received for {len(predictions)} objects.")
            self._redraw_canvas()

    def start(self): plt.show()
    
    def on_key(self, event):
        # Universal keys
        if event.key == 'n': self.callback('next_image')
        elif event.key == 'b': self.callback('prev_image')
        elif event.key in ['+', '=']: self.zoom(0.5)
        elif event.key == '-': self.zoom(2.0)
        
        # Pan logic (checks if zoomed in)
        # Using the first axis to check limits is sufficient due to shared axes
        xlim = self.ax_flat[0].get_xlim()
        is_zoomed = (xlim[1] - xlim[0]) < (self.img_shape[1] - 1)
        
        if event.key in ['up', 'down', 'left', 'right'] and is_zoomed: self.pan(event.key)

        # Mask-mode keys
        if self.mode == 'mask':
            if event.key == 'u' and self.in_label_mode: self.callback('classify_objects')
            elif event.key == 'v':
                if self.in_label_mode:
                    modes = ['labels', 'predictions', 'off']
                    self.label_view_mode = modes[(modes.index(self.label_view_mode) + 1) % len(modes)]
                    if self.label_view_mode == 'predictions' and not self.predictions: self.callback('predict_current')
                else:
                    modes = ['rings', 'solid', 'off']
                    self.mask_display_mode = modes[(modes.index(self.mask_display_mode) + 1) % len(modes)]
                self._redraw_canvas()
            elif event.key == 'l':
                self.in_label_mode = not self.in_label_mode
                self.label_view_mode = 'labels' if self.in_label_mode else 'off'
                if self.in_label_mode: self.callback('predict_current')
                self._redraw_canvas()
            elif event.key == 'x':
                self.masks.fill(0); self.object_labels.clear(); self.predictions.clear()
                self._redraw_canvas()
            elif event.key == 'm': self._toggle_merge_mode()
            elif event.key == 'f': self._toggle_draw_mode()
            elif event.key == 'enter' and self.in_merge_mode: self._finalize_merge()

    def _redraw_canvas(self, maintain_zoom: bool = True):
        if self.base_img is None: return
        
        # Save zoom state from the first axis (since all are shared)
        xlim, ylim = self.ax_flat[0].get_xlim(), self.ax_flat[0].get_ylim()

        # --- 1. Prepare Base Images for each panel ---
        # Panel 0: Composite (RGB)
        img_composite = self.base_img.copy()
        
        # Panel 1: Phase (Blue Ch, Index 2) -> Grayscale
        img_phase = self.base_img[:, :, 2]
        
        # Panel 2: Alive (Green Ch, Index 1) -> Grayscale (or colormap handled by imshow)
        img_alive = self.base_img[:, :, 1]
        
        # Panel 3: Dead (Red Ch, Index 0) -> Grayscale
        img_dead = self.base_img[:, :, 0]
        
        panels = [
            (self.ax_flat[0], img_composite, None),
            (self.ax_flat[1], img_phase, 'gray'),
            (self.ax_flat[2], img_alive, 'gray'), # 'gray' shows intensity best, can use 'viridis'
            (self.ax_flat[3], img_dead, 'gray')
        ]

        # --- 2. Draw Loop ---
        for ax, img, cmap in panels:
            ax.clear()
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            
            # --- Draw Overlays (Masks/Points) on ALL panels ---
            if self.mode == 'mask':
                if self.in_label_mode and self.label_view_mode != 'off':
                    source_dict = self.predictions if self.label_view_mode == 'predictions' else self.object_labels
                    for mask_id in self.object_labels.keys():
                        if mask_id == 0: continue
                        label_id = source_dict.get(mask_id, 0)
                        label_name = self.LABEL_MAP.get(label_id, 'unlabelled')
                        # Use RGB color for composite, but maybe simpler colors for grayscale views?
                        # Using same color map for consistency
                        color = self.COLOR_MAP.get(label_name, self.COLOR_MAP['unlabelled'])
                        boundaries = find_boundaries(self.masks == mask_id, mode='inner')
                        self._draw_boundaries(ax, boundaries, color)
                        
                elif self.masks.max() > 0 and self.mask_display_mode != 'off':
                    if self.mask_display_mode == 'rings':
                        boundaries = find_boundaries(self.masks, mode='inner')
                        # Yellow rings for visibility across all backgrounds
                        self._draw_boundaries(ax, boundaries, [1, 1, 0]) 
                    elif self.mask_display_mode == 'solid':
                        # Solid overlay is tricky on grayscale, might obscure data. 
                        # We apply it lightly.
                        self._generate_solid_mask_overlay(ax, img)

            if self.in_merge_mode:
                for lbl in self.merge_candidates:
                    mask_region = self.masks == lbl
                    self._draw_highlight(ax, mask_region)

            # --- Draw Points ---
            if self.mode == 'point':
                for x, y, label_id in self.point_labels:
                    label_name = self.POINT_LABEL_MAP.get(label_id)
                    color = self.POINT_COLOR_MAP.get(label_name)
                    ax.plot(x, y, marker='x', color=color, markersize=8, mew=2)

        # Restore State
        self._update_help_text()
        
        # Restore titles (cleared by ax.clear)
        titles = ["Composite (RGB)", "Phase (Structure)", "Alive (Green Ch)", "Dead (Red Ch)"]
        for ax, t in zip(self.ax_flat, titles):
            ax.set_title(t, fontsize=10, fontweight='bold')
            ax.axis('off')

        if maintain_zoom: 
            self.ax_flat[0].set_xlim(xlim)
            self.ax_flat[0].set_ylim(ylim)
        else:
            self.ax_flat[0].set_xlim(0, self.img_shape[1])
            self.ax_flat[0].set_ylim(self.img_shape[0], 0)
            
        self.fig.canvas.draw_idle()

    def _draw_boundaries(self, ax, boundaries, color):
        # Helper to draw boundaries on a specific axis
        # Create an RGBA overlay
        h, w = boundaries.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        overlay[boundaries] = list(color) + [1.0] # Solid alpha
        ax.imshow(overlay)

    def _draw_highlight(self, ax, mask_region):
        h, w = mask_region.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        overlay[mask_region] = [1, 1, 0, 0.3] # Yellow tint
        ax.imshow(overlay)

    def _generate_solid_mask_overlay(self, ax, base_img):
        # Simplified solid overlay using alpha blending
        overlay = np.zeros((self.img_shape[0], self.img_shape[1], 4), dtype=np.float32)
        for label in np.unique(self.masks):
            if label == 0: continue
            if label not in self.color_cache: self.color_cache[label] = np.random.rand(3)
            mask_region = self.masks == label
            overlay[mask_region] = list(self.color_cache[label]) + [0.3] # 0.3 Alpha
        ax.imshow(overlay)

    def on_click(self, event):
        # Allow clicking on ANY axis
        if event.inaxes not in self.ax_flat or event.xdata is None: return
        x, y = int(round(event.xdata)), int(round(event.ydata))

        # Point Mode
        if self.mode == 'point':
            point_to_delete = -1
            for i, (px, py, _label) in enumerate(self.point_labels):
                if abs(x - px) < 10 and abs(y - py) < 10: # Increased tolerance
                    point_to_delete = i
                    break
            if point_to_delete != -1: del self.point_labels[point_to_delete]
            elif event.button == 1: self.point_labels.append((x, y, 1))
            elif event.button == 3: self.point_labels.append((x, y, 2))
            self._redraw_canvas()
            return

        # Mask Mode
        lbl = self.masks[y, x]
        if self.in_label_mode and event.button == 1 and lbl != 0:
            current_label = self.object_labels.get(lbl, 0)
            next_label = (current_label + 1) % len(self.LABEL_MAP)
            self.object_labels[lbl] = next_label
            if self.label_view_mode != 'off': self._redraw_canvas()
        elif event.button == 3 and lbl != 0:
            self.masks[self.masks == lbl] = 0
            if lbl in self.object_labels: del self.object_labels[lbl]
            if lbl in self.predictions: del self.predictions[lbl]
            self._redraw_canvas()
        elif event.button == 1:
            if self.in_merge_mode and lbl != 0 and lbl not in self.merge_candidates:
                self.merge_candidates.append(lbl)
                self._redraw_canvas()
            elif self.in_draw_mode:
                self.is_actively_drawing = True
                self.active_ax = event.inaxes # Remember which panel started the draw
                self.draw_points = [(x, y)]
                # Create polygon artist ONLY on the active panel for performance during drag
                self.draw_artist = Polygon(self.draw_points, animated=True, closed=False, edgecolor='r', linewidth=1, fill=False)
                self.active_ax.add_patch(self.draw_artist)
                self.background = self.fig.canvas.copy_from_bbox(self.active_ax.bbox)
                self.fig.canvas.draw()
                
    def on_draw(self, event): 
        # Only capture background if we are actively drawing
        if self.is_actively_drawing and self.active_ax:
             self.background = self.fig.canvas.copy_from_bbox(self.active_ax.bbox)

    def on_release(self, event):
        if not self.is_actively_drawing: return
        self.is_actively_drawing = False
        self._finalize_drawing()

    def on_motion(self, event):
        # Only allow drawing on the axis we started on
        if not self.is_actively_drawing or event.inaxes != self.active_ax or event.xdata is None: return
        x, y = int(event.xdata), int(event.ydata)
        self.draw_points.append((x, y))
        self.draw_artist.set_xy(self.draw_points)
        if self.background is None: return
        
        # Blit optimization
        self.fig.canvas.restore_region(self.background)
        self.active_ax.draw_artist(self.draw_artist)
        self.fig.canvas.blit(self.active_ax.bbox)

    def _toggle_merge_mode(self):
        self.in_merge_mode = not self.in_merge_mode; self.merge_candidates = []
        self._redraw_canvas()
    def _toggle_draw_mode(self): self.in_draw_mode = not self.in_draw_mode

    def _finalize_merge(self):
        if len(self.merge_candidates) > 1:
            target_lbl = self.merge_candidates[0]
            for lbl in self.merge_candidates[1:]: self.masks[self.masks == lbl] = target_lbl
        self._toggle_merge_mode()

    def _finalize_drawing(self):
        if len(self.draw_points) > 2:
            xs, ys = [p[0] for p in self.draw_points], [p[1] for p in self.draw_points]
            if max(xs) - min(xs) > 1 and max(ys) - min(ys) > 1:
                rr, cc = draw_polygon(ys, xs, self.img_shape)
                new_label = self.masks.max() + 1
                self.masks[rr, cc] = new_label
                if self.in_label_mode: self.object_labels[new_label] = 0
        
        if self.draw_artist: self.draw_artist.remove()
        self.draw_points = []; self.draw_artist = None; self.active_ax = None
        self._redraw_canvas()

    def zoom(self, factor: float):
        # Apply zoom to the first axis; shared axes handle the rest
        ax = self.ax_flat[0]
        x, y = ax.get_xlim(), ax.get_ylim()
        cx, cy = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        new_w, new_h = (x[1] - x[0]) * factor, (y[1] - y[0]) * factor
        ax.set_xlim(cx - new_w / 2, cx + new_w / 2)
        ax.set_ylim(cy - new_h / 2, cy + new_h / 2)
        self.fig.canvas.draw_idle()

    def pan(self, direction: str):
        ax = self.ax_flat[0]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        width, height = xlim[1] - xlim[0], ylim[0] - ylim[1]
        dx = dy = 0; pan_factor = 0.1
        if direction == 'up': dy = -height * pan_factor
        if direction == 'down': dy = height * pan_factor
        if direction == 'left': dx = -width * pan_factor
        if direction == 'right': dx = width * pan_factor
        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.fig.canvas.draw_idle()

    def _update_help_text(self):
        if self.help_text_artist: self.help_text_artist.remove()
        text = "Controls: Zoom (+/-), Pan (Arrows), Next/Prev (n/b). "
        if self.mode == 'point': text += "Left/Right Click: Add Point"
        elif self.in_label_mode: text += "Left-Click: Cycle Label, 'v': View"
        else: text += "'f': Draw, 'm': Merge, 'x': Clear"
        self.help_text_artist = self.fig.text(0.01, 0.99, text, transform=self.fig.transFigure, fontsize=9, bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
