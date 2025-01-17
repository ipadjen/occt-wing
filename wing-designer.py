import sys
import numpy as np
import ssl
import urllib.request as urllib2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, QGridLayout,
                            QFileDialog, QTabWidget, QSplitter, QMessageBox, QComboBox,
                            QProgressDialog, QCheckBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from OCC.Display.backend import load_backend
load_backend('pyqt5')
from OCC.Display.qtDisplay import qtViewer3d
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Ax2, gp_Dir
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeWire,
                                    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Transform)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone

class WingGenerator:
    def __init__(self):
        self.common_airfoils = [
            's1223', 'e423', 'clarky', 'n0012', 'naca2412', 'naca4412',
            'naca23012', 'raf15', 'rae2822', 'sd7062'
        ]

    def generate_naca4(self, code, num_points=100):
        """Generate NACA 4 series airfoil points"""
        try:
            if not code.isdigit() or len(code) != 4:
                raise ValueError("Invalid NACA code format")

            m = int(code[0]) / 100.0
            p = int(code[1]) / 10.0
            t = int(code[2:]) / 100.0

            x = np.linspace(0, 1, num_points)
            y_c = np.zeros_like(x)
            y_t = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                          0.2843 * x**3 - 0.1015 * x**4)

            for i in range(len(x)):
                if x[i] < p:
                    y_c[i] = m * (x[i] / p**2) * (2 * p - x[i])
                else:
                    y_c[i] = m * ((1 - x[i]) / (1 - p)**2) * (1 + x[i] - 2 * p)

            theta = np.arctan(np.gradient(y_c, x))
            x_u = x - y_t * np.sin(theta)
            x_l = x + y_t * np.sin(theta)
            y_u = y_c + y_t * np.cos(theta)
            y_l = y_c - y_t * np.cos(theta)

            return np.vstack((np.hstack((x_u[::-1], x_l[1:])),
                            np.hstack((y_u[::-1], y_l[1:])))).T

        except Exception as e:
            raise Exception(f"Failed to generate NACA airfoil: {str(e)}")

    def fetch_uiuc_airfoil(self, profile):
        """Fetch airfoil coordinates from UIUC database"""
        try:
            foil_dat_url = f"http://m-selig.ae.illinois.edu/ads/coord_seligFmt/{profile}.dat"
            ssl._create_default_https_context = ssl._create_unverified_context

            response = urllib2.urlopen(foil_dat_url)
            points = []

            for line in response.readlines()[1:]:  # Skip header
                try:
                    data = line.decode().strip().split()
                    if len(data) == 2:
                        x, y = map(float, data)
                        points.append([x, y])
                except ValueError:
                    continue

            if not points:
                raise Exception("No valid points found in airfoil data")

            return np.array(points)

        except Exception as e:
            raise Exception(f"Failed to fetch airfoil data: {str(e)}")

    def read_airfoil_dat(self, filename):
        """Read airfoil coordinates from dat file"""
        try:
            points = []
            with open(filename, 'r') as f:
                lines = f.readlines()

                # Skip header lines
                start_line = 0
                for i, line in enumerate(lines):
                    try:
                        values = [float(val) for val in line.strip().split()]
                        if len(values) == 2:
                            start_line = i
                            break
                    except ValueError:
                        continue

                # Read coordinates
                for line in lines[start_line:]:
                    try:
                        x, y = map(float, line.strip().split())
                        points.append([x, y])
                    except ValueError:
                        continue

            if not points:
                raise Exception("No valid points found in file")

            points = np.array(points)

            # Normalize if necessary
            x_max = np.max(points[:, 0])
            x_min = np.min(points[:, 0])
            if abs(x_max - 1.0) > 0.01 or abs(x_min) > 0.01:
                chord_length = x_max - x_min
                points[:, 0] = (points[:, 0] - x_min) / chord_length
                points[:, 1] = points[:, 1] / chord_length

            return points

        except Exception as e:
            raise Exception(f"Failed to read airfoil file: {str(e)}")

    def create_wing(self, root_airfoil, span, sweep_angle=0.0, taper_ratio=1.0,
                    twist_angle=0.0, tip_airfoil=None, chord_root=1.0, te_tolerance=0.001,
                    num_sections=10, blend_start_section=7, ellipse=0):
        """
        Create a wing with multiple sections and gradual airfoil blending

        Some parameters:
        num_sections=10 - number of sections for loft operation
        blend_start_section=0.7 - percentage of span when blending between
                                  different airfoil starts
        ellipse - 0 for no ellipse, 1 for half ellipse, 2 for full ellipse
        """
        try:
            # Process root airfoil
            if isinstance(root_airfoil, str) and len(root_airfoil) == 4:
                root_points = self.generate_naca4(root_airfoil)
            else:
                root_points = np.array(root_airfoil)
    
            # Process tip airfoil
            if tip_airfoil is None:
                tip_points = root_points.copy()
            elif isinstance(tip_airfoil, str) and len(tip_airfoil) == 4:
                tip_points = self.generate_naca4(tip_airfoil)
            else:
                tip_points = np.array(tip_airfoil)
    
            # Process trailing edges for root and tip airfoils
            root_points = self.process_trailing_edge(root_points, te_tolerance)
            tip_points = self.process_trailing_edge(tip_points, te_tolerance)
    
            # Ensure both airfoils have the same number of points after processing
            if len(root_points) != len(tip_points):
                target_length = min(len(root_points), len(tip_points))
                root_points = self.resample_airfoil(root_points, target_length)
                tip_points = self.resample_airfoil(tip_points, target_length)
    
            # Create sections
            generator = BRepOffsetAPI_ThruSections(True, True)
            generator.CheckCompatibility(False)
    
            for i in range(num_sections):
                x_offset = 0 # Initialize x_offset
                # Calculate section parameters
                section_ratio = i / (num_sections - 1)
                z_pos = span * section_ratio
                # Calculate spanwise position relative to root (0 to 1)
                z_rel = z_pos / span

                if ellipse == 0:
                    chord = chord_root * (1 - section_ratio * (1 - taper_ratio))
                else:
                    tip_chord = chord_root * taper_ratio
                    if ellipse == 1: # Half ellipse
                        if section_ratio < 0.99:  # Allow some margin before tip
                            min_chord = tip_chord
                            chord_range = chord_root - min_chord
                            elliptical_ratio = np.sqrt(1 - (z_rel * z_rel))
                            chord = min_chord + (chord_range * elliptical_ratio)
                        else:
                            chord = tip_chord
                        x_offset = 0
                    else: # Full ellipse
                        chord = chord_root * np.sqrt(1 - (z_rel * z_rel))
                        chord = max(chord, tip_chord)
                        x_offset = (chord_root - chord) / 2

                # Calculate blending factor for airfoil shape
                if z_rel >= blend_start_section:
                    blend_ratio = (z_rel - blend_start_section) / (1.0 - blend_start_section + 1e-7)
                    blend_ratio = min(1.0, max(0.0, blend_ratio))
                    section_points = (1 - blend_ratio) * root_points + blend_ratio * tip_points
                else:
                    section_points = root_points
    
                # Create section wire
                section_wire = self.create_section_wire(section_points, scale=chord)
    
                # Create transformation
                transform = gp_Trsf()
    
                # 1. Move section to span position
                self.add_translation(transform, gp_Vec(0, 0, z_pos))

                # 2. Apply x translation (sweep or ellipse movement)
                if ellipse == 0:
                    sweep_rad = np.radians(sweep_angle)
                    x_offset = z_pos * np.tan(sweep_rad)
                self.add_translation(transform, gp_Vec(x_offset, 0, 0))

                # 3. Apply twist around quarter chord
                if abs(twist_angle) > 0.001:
                    quarter_chord = chord * 0.25
    
                    # Apply twist around quarter chord
                    twist_rad = np.radians(twist_angle * section_ratio)
                    self.add_rotation(transform, twist_rad, gp_Dir(0, 0, 1),
                                    gp_Pnt(quarter_chord, 0, 0))

                # Transform and add section
                transformed_section = BRepBuilderAPI_Transform(section_wire, transform).Shape()
                generator.AddWire(transformed_section)
    
            generator.Build()

            if not generator.IsDone():
                raise RuntimeError("Failed to create wing")
    
            return generator.Shape()
    
        except Exception as e:
           raise Exception(f"Failed to create wing: {str(e)}")
    
    def process_trailing_edge(self, points, te_tolerance):
        """Process trailing edge of airfoil points"""
        try:
            points = points.copy()  # Work with a copy to preserve original
            original_point_count = len(points)
            MIN_POINTS = 20
    
            while len(points) > MIN_POINTS:
                first_point = points[0]
                last_point = points[-1]
                trailing_edge_gap = (last_point[0] - first_point[0])**2 + \
                                    (last_point[1] - first_point[1])**2
    
                if trailing_edge_gap < te_tolerance**2:
                    points = points[1:-1]
                else:
                    break
    
            if len(points) <= MIN_POINTS:
                raise RuntimeError(
                    f"Reached minimum point limit ({MIN_POINTS}) without finding "
                    f"suitable trailing edge configuration"
                )
    
            return points
    
        except Exception as e:
            raise Exception(f"Failed to process trailing edge: {str(e)}")
    
    def create_section_wire(self, points, scale=1.0):
        """Create a wire for a wing section without trailing edge check"""
        try:
            # Scale points
            scaled_points = points.copy() * scale
    
            # Create array of points
            array = TColgp_Array1OfPnt(1, len(scaled_points))
            for i, point in enumerate(scaled_points, 1):
                array.SetValue(i, gp_Pnt(point[0], point[1], 0))
    
            # Create B-spline
            spline = GeomAPI_PointsToBSpline(array)
            if not spline.IsDone():
                raise RuntimeError("Failed to create B-spline")
    
            # Create edge from spline
            spline_edge = BRepBuilderAPI_MakeEdge(spline.Curve()).Edge()
    
            # Make wire
            wire_maker = BRepBuilderAPI_MakeWire()
            wire_maker.Add(spline_edge)
    
            # Create trailing edge
            first_point = gp_Pnt(scaled_points[0][0], scaled_points[0][1], 0)
            last_point = gp_Pnt(scaled_points[-1][0], scaled_points[-1][1], 0)
            trailing_edge = BRepBuilderAPI_MakeEdge(first_point, last_point).Edge()
            wire_maker.Add(trailing_edge)
    
            if not wire_maker.IsDone():
                raise RuntimeError("Failed to create wire")
    
            return wire_maker.Wire()
    
        except Exception as e:
            raise Exception(f"Failed to create section wire: {str(e)}")

    def resample_airfoil(self, points, n_points):
        """Resample airfoil points to have exactly n_points"""
        try:
            # Calculate cumulative distance along the airfoil
            distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
            distances = np.insert(distances, 0, 0)

            # Normalize distances
            distances = distances / distances[-1]

            # Create new evenly spaced points
            new_distances = np.linspace(0, 1, n_points)

            # Interpolate x and y coordinates
            new_x = np.interp(new_distances, distances, points[:, 0])
            new_y = np.interp(new_distances, distances, points[:, 1])

            return np.column_stack((new_x, new_y))

        except Exception as e:
            raise Exception(f"Failed to resample airfoil: {str(e)}")

    def add_translation(self, transform_matrix, transform_vec):
        """Add translation to transformation matrix"""
        try:
            translation = gp_Trsf()
            translation.SetTranslation(transform_vec)
            transform_matrix.Multiply(translation)
            return transform_matrix

        except Exception as e:
            raise Exception(f"Failed to add translation: {str(e)}")

    def add_rotation(self, transform_matrix, rotation_angle, rotation_dir, rotation_pt=gp_Pnt(0, 0, 0)):
        """Add rotation to transformation matrix"""
        try:
            rotation = gp_Trsf()
            rotation.SetRotation(gp_Ax1(rotation_pt, rotation_dir), rotation_angle)
            transform_matrix.Multiply(rotation)
            return transform_matrix

        except Exception as e:
            raise Exception(f"Failed to add rotation: {str(e)}")

    def export_step(self, shape, filename):
        """Export shape to STEP file"""
        try:
            writer = STEPControl_Writer()
            Interface_Static.SetCVal("write.step.schema", "AP214")
            Interface_Static.SetCVal("write.step.unit","M")

            status = writer.Transfer(shape, STEPControl_AsIs)
            if status != IFSelect_RetDone:
                raise Exception("Failed to transfer shape")

            status = writer.Write(filename)
            if status != IFSelect_RetDone:
                raise Exception("Failed to write file")

            return True

        except Exception as e:
            raise Exception(f"Failed to export STEP file: {str(e)}")

class WingDesignerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wing Designer")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize wing generator
        self.wing_generator = WingGenerator()
        self.current_wing = None

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)

        # Create left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Create airfoil sections
        self.setup_airfoil_sections(left_layout)

        # Add other parameters
        self.setup_parameters(left_layout)

        # Add buttons
        self.setup_buttons(left_layout)

        # Create right panel for viewers
        right_panel = QWidget()
        right_layout = QGridLayout(right_panel)

        # Setup viewers
        self.setup_viewers(right_layout)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Set initial sizes
        splitter.setSizes([300, 900])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

    def setup_airfoil_sections(self, layout):
        """Setup root and tip airfoil sections"""
        # Root airfoil section
        root_group = QWidget()
        root_layout = QVBoxLayout(root_group)
        root_layout.addWidget(QLabel("Root Airfoil"))
        self.root_tabs = self.create_airfoil_tabs('root')
        root_layout.addWidget(self.root_tabs)
        layout.addWidget(root_group)

        # Tip airfoil section
        tip_group = QWidget()
        tip_layout = QVBoxLayout(tip_group)
        tip_layout.addWidget(QLabel("Tip Airfoil"))
        self.tip_tabs = self.create_airfoil_tabs('tip')
        tip_layout.addWidget(self.tip_tabs)
        layout.addWidget(tip_group)

    def create_airfoil_tabs(self, section):
        """Create tabs for different airfoil input methods"""
        tabs = QTabWidget()

        # NACA 4 tab
        naca_tab = QWidget()
        naca_layout = QVBoxLayout(naca_tab)  # Changed to VBoxLayout
        naca_input_layout = QHBoxLayout()
        naca_input = QLineEdit('0012')
        naca_input_layout.addWidget(QLabel("NACA"))
        naca_input_layout.addWidget(naca_input)
        naca_layout.addLayout(naca_input_layout)
        naca_preview_btn = QPushButton("Preview")
        naca_preview_btn.clicked.connect(lambda: self.preview_airfoil(section))
        naca_layout.addWidget(naca_preview_btn)

        # Import tab
        import_tab = QWidget()
        import_layout = QVBoxLayout(import_tab)  # Changed to VBoxLayout
        file_layout = QHBoxLayout()
        file_path = QLineEdit()
        file_path.setReadOnly(True)
        file_layout.addWidget(file_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_airfoil_file(section))
        file_layout.addWidget(browse_btn)
        import_layout.addLayout(file_layout)
        import_preview_btn = QPushButton("Preview")
        import_preview_btn.clicked.connect(lambda: self.preview_airfoil(section))
        import_layout.addWidget(import_preview_btn)

        # Database tab
        db_tab = QWidget()
        db_layout = QVBoxLayout(db_tab)
        db_combo = QComboBox()
        db_combo.addItems(self.wing_generator.common_airfoils)
        db_combo.setEditable(True)
        db_combo.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
        db_layout.addWidget(QLabel("Enter or select airfoil:"))
        db_layout.addWidget(db_combo)
        db_preview_btn = QPushButton("Preview")
        db_preview_btn.clicked.connect(lambda: self.preview_database_airfoil(section))
        db_layout.addWidget(db_preview_btn)

        tabs.addTab(naca_tab, "NACA 4")
        tabs.addTab(import_tab, "Import")
        tabs.addTab(db_tab, "Database")

        # Store references to widgets
        setattr(self, f"{section}_naca_input", naca_input)
        setattr(self, f"{section}_file_path", file_path)
        setattr(self, f"{section}_db_combo", db_combo)

        return tabs

    def setup_parameters(self, layout):
        """Setup parameter input fields"""
        self.inputs = {}
    
        # Add wing type selector
        type_group = QWidget()
        type_layout = QHBoxLayout(type_group)
        type_layout.addWidget(QLabel("Wing Type:"))
        self.wing_type_combo = QComboBox()
        self.wing_type_combo.addItems(["Straight", "Tapered", "Swept", "Elliptical"])
        self.wing_type_combo.currentTextChanged.connect(self.update_parameter_visibility)
        type_layout.addWidget(self.wing_type_combo)
        layout.addWidget(type_group)
    
        # Create parameter groups
        self.param_widgets = {}
    
        # Common parameters for all wing types
        self.common_params = {
            'span': ('Half Wing Span (m)', '10.0'),
            'chord_root': ('Root Chord (m)', '2.0'),
            'twist_angle': ('Twist Angle (deg)', '2.0'),
            'te_tolerance': ('Trailing Edge Tolerance (m)', '0.001')
        }
    
        # Tapered wing specific parameters
        self.tapered_params = {
            'tapered_tip_chord': ('Tip Chord (m)', '1.2'),
        }
    
        # Swept wing specific parameters
        self.swept_params = {
            'le_sweep': ('Leading Edge Sweep (deg)', '15.0'),
            'swept_tip_chord': ('Tip Chord (m)', '1.2')
        }
    
        # Elliptical wing specific parameters
        self.elliptical_params = {
            'elliptical_tip_chord': ('Tip Chord (m)', '0.1')
        }
    
        # Create all parameter widgets
        params_group = QWidget()
        self.params_layout = QGridLayout(params_group)
        row = 0
    
        # Add common parameters
        for key, (label, default) in self.common_params.items():
            label_widget = QLabel(label)
            input_widget = QLineEdit(default)
            self.params_layout.addWidget(label_widget, row, 0)
            self.params_layout.addWidget(input_widget, row, 1)
            self.inputs[key] = input_widget
            self.param_widgets[f'common_{key}'] = (label_widget, input_widget)
            row += 1
    
        # Add "Full Wing" checkbox
        full_wing_group = QWidget()
        full_wing_layout = QHBoxLayout(full_wing_group)
        self.full_wing_checkbox = QCheckBox("Full Wing")
        full_wing_layout.addWidget(self.full_wing_checkbox)
        self.params_layout.addWidget(full_wing_group, row, 0, 1, 2)
        row += 1
    
        # Add tapered parameters
        for key, (label, default) in self.tapered_params.items():
            label_widget = QLabel(label)
            input_widget = QLineEdit(default)
            self.params_layout.addWidget(label_widget, row, 0)
            self.params_layout.addWidget(input_widget, row, 1)
            self.inputs[key] = input_widget
            self.param_widgets[f'tapered_{key}'] = (label_widget, input_widget)
            row += 1
    
        # Add taper checkboxes
        self.tapered_checkbox_group = QWidget()
        tapered_checkbox_layout = QHBoxLayout(self.tapered_checkbox_group)
        self.le_taper = QCheckBox("Leading Edge Taper")
        self.te_taper = QCheckBox("Trailing Edge Taper")
        tapered_checkbox_layout.addWidget(self.le_taper)
        tapered_checkbox_layout.addWidget(self.te_taper)
        self.params_layout.addWidget(self.tapered_checkbox_group, row, 0, 1, 2)
        self.param_widgets['tapered_checkboxes'] = (self.tapered_checkbox_group,)
        row += 1
    
        # Add swept parameters
        for key, (label, default) in self.swept_params.items():
            label_widget = QLabel(label)
            input_widget = QLineEdit(default)
            self.params_layout.addWidget(label_widget, row, 0)
            self.params_layout.addWidget(input_widget, row, 1)
            self.inputs[key] = input_widget
            self.param_widgets[f'swept_{key}'] = (label_widget, input_widget)
            row += 1
    
        # Add elliptical parameters
        for key, (label, default) in self.elliptical_params.items():
            label_widget = QLabel(label)
            input_widget = QLineEdit(default)
            self.params_layout.addWidget(label_widget, row, 0)
            self.params_layout.addWidget(input_widget, row, 1)
            self.inputs[key] = input_widget
            self.param_widgets[f'elliptical_{key}'] = (label_widget, input_widget)
            row += 1
    
        # Add elliptical checkbox
        self.elliptical_checkbox_group = QWidget()
        elliptical_checkbox_layout = QHBoxLayout(self.elliptical_checkbox_group)
        self.full_elliptical = QCheckBox("Full Elliptical (both edges)")
        elliptical_checkbox_layout.addWidget(self.full_elliptical)
        self.params_layout.addWidget(self.elliptical_checkbox_group, row, 0, 1, 2)
        self.param_widgets['elliptical_checkboxes'] = (self.elliptical_checkbox_group,)
        row += 1
    
        layout.addWidget(params_group)
    
        # Initialize parameter visibility
        self.update_parameter_visibility(self.wing_type_combo.currentText())

    def update_parameter_visibility(self, wing_type):
        """Update visible parameters based on wing type"""
        try:
            # Show common parameters for all wing types
            for key in self.common_params.keys():
                label, input_widget = self.param_widgets[f'common_{key}']
                label.show()
                input_widget.show()
    
            # Hide all specific parameters first
            for key in self.tapered_params.keys():
                if f'tapered_{key}' in self.param_widgets:
                    label, input_widget = self.param_widgets[f'tapered_{key}']
                    label.hide()
                    input_widget.hide()
    
            for key in self.swept_params.keys():
                if f'swept_{key}' in self.param_widgets:
                    label, input_widget = self.param_widgets[f'swept_{key}']
                    label.hide()
                    input_widget.hide()
    
            for key in self.elliptical_params.keys():
                if f'elliptical_{key}' in self.param_widgets:
                    label, input_widget = self.param_widgets[f'elliptical_{key}']
                    label.hide()
                    input_widget.hide()
    
            # Hide all checkboxes first
            self.tapered_checkbox_group.hide()
            self.elliptical_checkbox_group.hide()
    
            # Show parameters based on wing type
            if wing_type == "Tapered":
                # Show tapered wing specific parameters
                for key in self.tapered_params.keys():
                    if f'tapered_{key}' in self.param_widgets:
                        label, input_widget = self.param_widgets[f'tapered_{key}']
                        label.show()
                        input_widget.show()
                self.tapered_checkbox_group.show()
    
            elif wing_type == "Swept":
                # Show swept wing specific parameters
                for key in self.swept_params.keys():
                    if f'swept_{key}' in self.param_widgets:
                        label, input_widget = self.param_widgets[f'swept_{key}']
                        label.show()
                        input_widget.show()
    
            elif wing_type == "Elliptical":
                # Show elliptical wing specific parameters
                for key in self.elliptical_params.keys():
                    if f'elliptical_{key}' in self.param_widgets:
                        label, input_widget = self.param_widgets[f'elliptical_{key}']
                        label.show()
                        input_widget.show()
                self.elliptical_checkbox_group.show()
    
            # Adjust layout
            self.params_layout.update()
            self.update()
    
        except Exception as e:
            self.show_message("Error", f"Failed to update parameters: {str(e)}", True)

    def setup_buttons(self, layout):
        """Setup action buttons"""
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
    
        generate_btn = QPushButton("Generate Wing")
        generate_btn.clicked.connect(self.generate_wing)
        buttons_layout.addWidget(generate_btn)
    
        export_step_btn = QPushButton("Export STEP")
        export_step_btn.clicked.connect(self.export_step)
        buttons_layout.addWidget(export_step_btn)
    
        layout.addWidget(buttons_widget)
        layout.addStretch()

    def setup_viewers(self, layout):
        """Setup 2D and 3D viewers"""
        # Create matplotlib figures for 2D airfoils
        self.root_figure = Figure(figsize=(5, 3))
        self.tip_figure = Figure(figsize=(5, 3))
        self.root_canvas = FigureCanvas(self.root_figure)
        self.tip_canvas = FigureCanvas(self.tip_figure)

        # Create 3D viewer for wing
        self.wing_viewer = qtViewer3d(self)
        self.wing_viewer._display.display_triedron()
        self.wing_viewer._display.View.SetProj(1, 1, 1)

        # Add viewers to layout
        layout.addWidget(self.root_canvas, 0, 0)
        layout.addWidget(self.tip_canvas, 0, 1)
        layout.addWidget(self.wing_viewer, 1, 0, 1, 2)

        # Set row stretch factors
        layout.setRowStretch(0, 33)
        layout.setRowStretch(1, 66)

    def show_message(self, title, message, is_error=False):
        """Show popup message dialog"""
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Critical if is_error else QMessageBox.Information)
        msg.exec_()

    def browse_airfoil_file(self, section):
        """Open file dialog to browse for airfoil dat file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {section} airfoil file",
            "",
            "Airfoil data files (*.dat);;All files (*.*)"
        )
        if filename:
            file_path = getattr(self, f"{section}_file_path")
            file_path.setText(filename)
            self.preview_airfoil(section)

    def preview_database_airfoil(self, section):
        """Preview airfoil from database"""
        try:
            progress = QProgressDialog("Fetching airfoil data...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            combo = getattr(self, f"{section}_db_combo")
            profile = combo.currentText()
            points = self.wing_generator.fetch_uiuc_airfoil(profile)

            progress.close()

            if points is not None:
                self.plot_airfoil(section, points, f"Database Airfoil: {profile}")
            else:
                self.show_message("Error", f"Failed to load airfoil {profile}", True)

        except Exception as e:
            progress.close()
            self.show_message("Error", str(e), True)

    def preview_airfoil(self, section):
        """Preview airfoil based on current tab selection"""
        try:
            tabs = getattr(self, f"{section}_tabs")
            
            if tabs.currentIndex() == 0:  # NACA
                naca_input = getattr(self, f"{section}_naca_input")
                naca_code = naca_input.text()
                points = self.wing_generator.generate_naca4(naca_code)
                if points is not None:
                    self.plot_airfoil(section, points, f"NACA {naca_code}")
                
            elif tabs.currentIndex() == 1:  # Import
                file_path = getattr(self, f"{section}_file_path")
                if not file_path.text():
                    raise Exception("No file selected")
                points = self.wing_generator.read_airfoil_dat(file_path.text())
                if points is not None:
                    self.plot_airfoil(section, points, f"Imported: {file_path.text().split('/')[-1]}")
    
        except Exception as e:
            self.show_message("Error", str(e), True)

    def plot_airfoil(self, section, points, title):
        """Plot airfoil points"""
        figure = self.root_figure if section == 'root' else self.tip_figure
        canvas = self.root_canvas if section == 'root' else self.tip_canvas
    
        figure.clear()
        ax = figure.add_subplot(111, aspect='equal')
        ax.plot(points[:, 0], points[:, 1], 'b-', label='Airfoil')
    
        # Adjust font sizes
        ax.set_title(title, fontsize=10)  # Smaller title
        ax.tick_params(axis='both', which='major', labelsize=7)  # Smaller tick labels
        ax.set_xlabel('X', fontsize=7)  # Smaller axis labels
        ax.set_ylabel('Y', fontsize=7)
    
        ax.grid(True)
    
        # Set reasonable axis limits
        margin = 0.1
        max_dim = max(np.max(points[:, 0]) - np.min(points[:, 0]),
                     np.max(points[:, 1]) - np.min(points[:, 1]))
        center_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2
        center_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
        ax.set_xlim(center_x - max_dim/2 - margin, center_x + max_dim/2 + margin)
        ax.set_ylim(center_y - max_dim/2 - margin, center_y + max_dim/2 + margin)
    
        # Adjust layout to ensure everything fits
        figure.tight_layout()
    
        canvas.draw()

    def get_airfoil_points(self, section):
        """Get airfoil points based on selected input method"""
        tabs = getattr(self, f"{section}_tabs")

        try:
            if tabs.currentIndex() == 0:  # NACA 4
                naca_input = getattr(self, f"{section}_naca_input")
                return self.wing_generator.generate_naca4(naca_input.text())

            elif tabs.currentIndex() == 1:  # Import
                file_path = getattr(self, f"{section}_file_path")
                if not file_path.text():
                    raise Exception("No file selected")
                return self.wing_generator.read_airfoil_dat(file_path.text())

            else:  # Database
                db_combo = getattr(self, f"{section}_db_combo")
                return self.wing_generator.fetch_uiuc_airfoil(db_combo.currentText())

        except Exception as e:
            raise Exception(f"Failed to get airfoil points: {str(e)}")

    def generate_wing(self):
        """Generate wing based on selected type and parameters"""
        try:
            progress = QProgressDialog("Generating wing...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
    
            wing_type = self.wing_type_combo.currentText()
    
            # Get common parameters
            half_span = float(self.inputs['span'].text())
            chord_root = float(self.inputs['chord_root'].text())
            twist_angle = float(self.inputs['twist_angle'].text())
            te_tolerance = float(self.inputs['te_tolerance'].text())
    
            # Get airfoil points
            root_points = self.get_airfoil_points('root')
            tip_points = self.get_airfoil_points('tip')
    
            if root_points is None or tip_points is None:
                raise Exception("Failed to get airfoil points")
    
            # Update matplotlib plots with final airfoil shapes
            self.plot_airfoil('root', root_points, "Root Airfoil")
            self.plot_airfoil('tip', tip_points, "Tip Airfoil")
    
            if wing_type == "Straight":
                self.current_wing = self.wing_generator.create_wing(
                    root_airfoil=root_points,
                    tip_airfoil=tip_points,
                    span=half_span,
                    taper_ratio=1.0,
                    twist_angle=twist_angle,
                    chord_root=chord_root,
                    te_tolerance=te_tolerance,
                    ellipse=0
                )
    
            elif wing_type == "Tapered":
                tip_chord = float(self.inputs['tapered_tip_chord'].text())
                taper_ratio = tip_chord / chord_root
    
                le_taper = self.le_taper.isChecked()
                te_taper = self.te_taper.isChecked()
    
                if (le_taper and te_taper) or (not le_taper and not te_taper):
                    sweep_offset = (chord_root - tip_chord) / 2
                    sweep_angle = np.degrees(np.arctan2(sweep_offset, half_span))
                elif le_taper:
                    sweep_offset = chord_root - tip_chord
                    sweep_angle = np.degrees(np.arctan2(sweep_offset, half_span))
                else:  # te_taper only
                    sweep_angle = 0
    
                self.current_wing = self.wing_generator.create_wing(
                    root_airfoil=root_points,
                    tip_airfoil=tip_points,
                    span=half_span,
                    sweep_angle=sweep_angle,
                    taper_ratio=taper_ratio,
                    twist_angle=twist_angle,
                    chord_root=chord_root,
                    te_tolerance=te_tolerance,
                    ellipse=0
                )
    
            elif wing_type == "Swept":
                tip_chord = float(self.inputs['swept_tip_chord'].text())
                le_sweep = float(self.inputs['le_sweep'].text())
    
                self.current_wing = self.wing_generator.create_wing(
                    root_airfoil=root_points,
                    tip_airfoil=tip_points,
                    span=half_span,
                    sweep_angle=le_sweep,
                    taper_ratio=tip_chord / chord_root,
                    twist_angle=twist_angle,
                    chord_root=chord_root,
                    te_tolerance=te_tolerance,
                    ellipse=0
                )
    
            elif wing_type == "Elliptical":
                tip_chord = float(self.inputs['elliptical_tip_chord'].text())
                full_elliptical = self.full_elliptical.isChecked()

                if full_elliptical:
                    ellipse=2
                else:
                    ellipse=1

                self.current_wing = self.wing_generator.create_wing(
                    root_airfoil=root_points,
                    tip_airfoil=tip_points,
                    span=half_span,
                    taper_ratio=tip_chord / chord_root,
                    twist_angle=twist_angle,
                    chord_root=chord_root,
                    te_tolerance=te_tolerance,
                    num_sections=40,
                    ellipse=ellipse
                )
    
            # Check if full wing is required
            if self.full_wing_checkbox.isChecked():
                # Create mirror transformation
                mirror_trsf = gp_Trsf()
                # Mirror about XZ plane (Y=0) instead of XY plane
                mirror_trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))) #Ax2 needed for symmetry
            
                # Create mirrored wing
                mirror_wing = BRepBuilderAPI_Transform(self.current_wing, mirror_trsf).Shape()

                # Fuse original and mirrored wings
                self.current_wing = BRepAlgoAPI_Fuse(self.current_wing, mirror_wing).Shape()
    
            # Update 3D viewer
            self.wing_viewer._display.EraseAll()
            self.wing_viewer._display.DisplayShape(self.current_wing, update=True)
            self.wing_viewer._display.FitAll()
    
            progress.close()
            self.show_message("Success", "Wing generated successfully!")
    
        except Exception as e:
            progress.close()
            self.show_message("Error", str(e), True)
    
    def export_step(self):
        """Export wing to STEP file"""
        if self.current_wing is None:
            self.show_message("Error", "No wing to export. Please generate a wing first.", True)
            return

        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save STEP file",
                "",
                "STEP files (*.step)"
            )
            if filename:
                if not filename.lower().endswith('.step'):
                    filename += '.step'

                progress = QProgressDialog("Exporting wing...", "Cancel", 0, 0, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()

                success = self.wing_generator.export_step(self.current_wing, filename)

                progress.close()

                if success:
                    self.show_message("Success", f"Wing exported to {filename}")
                else:
                    raise Exception("Failed to export wing")

        except Exception as e:
            self.show_message("Error", str(e), True)
    
def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show the main window
    window = WingDesignerGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
