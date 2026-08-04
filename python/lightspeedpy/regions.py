import numpy as np

class Region:
    """
    Class to store DS9 regions.
    
    Notes
    -----
    - All regions are assumed to be stored in `ciao` format.
    - All coordinates must be physical
    """
    def __init__(self, filename):
        """
        If you are looking for the region constructor, use `Region.load` instead. 
        """
        raise Exception("You cannot instantiate the pure base class Region")
    
    def load(filename):
        """
        Load a region from a region file

        Parameters
        ----------
        filename : str
            File name of the region

        Returns
        -------
        Region
            Region object
        """
        with open(filename) as f:
            line = f.readline()
            typ = line[:line.find('(')]
        if typ == "circle":
            return CircleRegion(filename)
        elif typ == "polygon":
            return PolygonRegion(filename)
        elif typ == "box":
            return BoxRegion(filename)
        elif typ == "ellipse":
            try:
                return EllipseAnnulusRegion(filename)
            except:
                return EllipseRegion(filename)
        elif typ == "annulus":
            return AnnulusRegion(filename)
        else:
            raise Exception(f"Unrecognized region type `{typ}`")
        
    def contains(self, x, y):
        """
        Return True if (x, y) is inside the region
        """
        raise Exception("You cannot instantiate the pure base class Region")

    def area(self):
        raise Exception("Area is only defined for non-polygon regions")

    
class EllipseAnnulusRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("ellipse"):
                raise Exception("Ellipse annulus regions must start with ellipse")
            if line.endswith('\n'): line = line[:-1]
            cx, cy, maj1, min1, maj2, min2, angle = line[8:-1].split(",")
            if ":" in cx:
                raise Exception("Can only load `ciao` formatted region files in physical coordinates")
            self.cx = float(cx)
            self.cy = float(cy)
            self.a1_2 = float(maj1)**2
            self.b1_2 = float(min1)**2
            self.a2_2 = float(maj2)**2
            self.b2_2 = float(min2)**2
            self.angle = float(angle) * np.pi / 180 - np.pi/2

    def contains(self, x, y):
        rot_x = np.sin(self.angle) * (x - self.cx) - np.cos(self.angle) * (y - self.cy)
        rot_y = np.cos(self.angle) * (x - self.cx) + np.sin(self.angle) * (y - self.cy)
        d1 = rot_x**2 / self.a1_2 + rot_y**2 / self.b1_2
        d2 = rot_x**2 / self.a2_2 + rot_y**2 / self.b2_2
        return (d1 > 1) & (d2 < 1)

    def area(self):
        return np.pi * (np.sqrt(self.a2_2 * self.b2_2) - np.sqrt(self.a1_2 * self.b1_2))
    
class AnnulusRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("annulus("):
                raise Exception("Annulus regions must start with annulus")
            if line.endswith('\n'): line = line[:-1]
            cx, cy, r1, r2 = line[8:-1].split(",")
            if ":" in cx:
                raise Exception("Can only load `ciao` formatted region files in physical coordinates")
            self.cx = float(cx)
            self.cy = float(cy)
            self.r1_2 = float(r1)**2
            self.r2_2 = float(r2)**2

    def contains(self, x, y):
        # Assume x and y are in degrees
        d2 = (x - self.cx)**2 + (y - self.cy)**2
        return (d2 > self.r1_2) & (d2 < self.r2_2)

    def area(self):
        return np.pi * (self.r2_2 - self.r1_2)

class BoxRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("box("):
                raise Exception("Box regions must start with box")
            if line.endswith('\n'): line = line[:-1]
            x, y, l, w, angle = line[4:-1].split(",")
            if ":" in x:
                raise Exception("Can only load `ciao` formatted region files in physical coordinates")
            self.x = float(x)
            self.y = float(y)
            self.l = float(l)
            self.w = float(w)
            self.angle = float(angle) * np.pi / 180

    def get_alpha(self, x, y, v0, v1):
        return ((x - self.x) * v0 + (y - self.y) * v1) / (v0*v0 + v1*v1)
    
    def contains(self, x, y):
        # Assume x and y are in degrees
        l_alpha = self.get_alpha(x, y, self.l * np.cos(self.angle), self.l * np.sin(self.angle))
        w_alpha = self.get_alpha(x, y, self.w * np.sin(self.angle), -self.w * np.cos(self.angle))
        return (np.abs(l_alpha) < 0.5) & (np.abs(w_alpha) < 0.5)

    def area(self):
        return self.l * self.w
  
      
class PolygonRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("polygon("):
                raise Exception("Polygon regions must start with polygon")
            if line.endswith('\n'): line = line[:-1]
            points = line[8:-1].split(",")
            self.points = []
            for i in range(0, len(points), 2):
                self.points.append((float(points[i]), float(points[i+1])))
            self.points = np.array(self.points)

    def contains_single(self, x, y):
        # Assume x and y are in degrees
        for line_index in range(len(self.points)):
            intersections_before = 1 # There is a self-intersection
            intersections_after = 0
            for other_line_index in range(len(self.points)):
                if other_line_index == line_index: continue
                # Check if the line intersects this line segment
                mid = (self.points[line_index] + self.points[line_index - 1])/2
                vx = x - mid[0]
                vy = y - mid[1]
                px, py = self.points[other_line_index-1] - self.points[other_line_index]
                diffx, diffy = mid - self.points[other_line_index]
                det = (vx * py - vy * px)
                alpha = (vx * diffy - vy * diffx) / det
                if 0 <= alpha <= 1:
                    beta = (px * diffy - py * diffx) / det
                    if beta > 1:
                        intersections_after += 1
                    else:
                        intersections_before += 1
            if intersections_before % 2 == 0:
                return False
            if intersections_after % 2 == 0:
                return False
        return True
    
    def contains(self, x, y):
        if type(x) == float:
            return self.contains_single(x, y)
        else:
            x_1d = x.reshape(-1)
            y_1d = y.reshape(-1)
            output = np.zeros(len(x_1d)).astype(bool)
            for i, (xi, yi) in enumerate(zip(x_1d, y_1d)):
                output[i] = self.contains_single(xi, yi)
            return output.reshape(x.shape)

class CircleRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("circle("):
                raise Exception("Circle regions must start with circle")
            if line.endswith('\n'): line = line[:-1]
            x, y, radius = line[7:-1].split(",")
            self.x = float(x)
            self.y = float(y)
            self.radius2 = float(radius)**2
    
    def contains(self, x, y):
        dist2 = (x - self.x)**2 + (y - self.y)**2
        return dist2 < self.radius2
    
    def area(self):
        return np.pi * self.radius2

    
class EllipseRegion(Region):
    def __init__(self, filename):
        with open(filename) as f:
            line = f.readline()
            if not line.startswith("ellipse("):
                raise Exception("Ellipse regions must start with ellipse")
            x, y, a, b, angle = line[8:-2].split(",")
            self.x = float(x)
            self.y = float(y)
            self.a = float(a)
            self.b = float(b)
            self.angle = float(angle) * np.pi / 180 - np.pi/2
    
    def contains(self, x, y):
        rot_x = np.sin(self.angle) * (x - self.x) - np.cos(self.angle) * (y - self.y)
        rot_y = np.cos(self.angle) * (x - self.x) + np.sin(self.angle) * (y - self.y)
        d = rot_x**2 / self.a**2 + rot_y**2 / self.b**2
        return d < 1
    
    def area(self):
        return np.pi * self.a * self.b
    