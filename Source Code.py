import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits import mplot3d
import math
import stl
import PySimpleGUI as sg
from decimal import Decimal

def BernsteinPoly(u,k,n):
    poly = [0 for i in range(len(u))]
    C = (math.factorial(n)) / math.factorial(k) / math.factorial(n - k)  # implement n choose k function
    for i in range(0, len(u)):
        poly[i] = C*u[i]**k*(1-u[i])**(n-k) #implement bernstein polynomial formula for each u value
    return poly  # return 1D array of bernstein polynomials for index k of degree  n

#Function 2: calculate bezier curve of any size
def BezierCurve(cp, num):
    rows,col = np.shape(cp)
    #calculate u values for number of points
    u = [0 for j in range(0, num)]
    for j in range(0, num):
        u[j] = 1/num*j
    polynomials = [0 for k in range(0, col)]
    for i in range(0,col):
        polynomials[i] = BernsteinPoly(u, i, col-1) #calculate bernstein polynomials for all u values
    points = [0 for i in range(0, num)]  # initialize points vector
    polynomials = np.transpose(polynomials)
    for i in range(0, num):
        points[i] = np.matmul(polynomials[i][:], np.transpose(cp)) #calculate each point value
    return points  # return 2D array of Bezier curve points

#Function 3: Bezier Surface Computation
def BezierSurface(cpp,num_u,num_v):
    u = [0 for j in range(0, num_u)]
    v = [0 for j in range(0, num_v)]
    for j in range(0, num_u):
        u[j] = 1/num_u * j
        v[j] = 1/num_v * j
    u_poly = [0 for k in range(0, 4)]
    v_poly = [0 for k in range(0, 4)]
    for i in range(0, 4):
        u_poly[i] = BernsteinPoly(u, i, 3)
        v_poly[i] = BernsteinPoly(v, i, 3)
    x = cpp[:][:][0]
    y = cpp[:][:][1]
    z = cpp[:][:][2]
    u_poly = np.transpose(u_poly)
    v_poly = np.transpose(v_poly)
    X = np.zeros((num_u, num_v))  # initialize data points
    Y = np.zeros((num_u, num_v))
    Z = np.zeros((num_u, num_v))
    for i in range(num_u):
        for j in range(num_v):
            ux = np.matmul(u_poly[i][:], x)  #multiply U*point
            uy = np.matmul(u_poly[i][:], y)
            uz = np.matmul(u_poly[i][:], z)
            X[i][j] = np.matmul(ux, v_poly[:][j])  #multiply U*point*V
            Y[i][j] = np.matmul(uy, v_poly[:][j])
            Z[i][j] = np.matmul(uz, v_poly[:][j])
    points = [X, Y, Z]  # return points in 3xUxV sized array
    return points

#Function 4: cubic Hermite curve computation, given 4 control points of 2d or 3d dimension
def HermiteCurve(cp,num):
    u = [0 for j in range(0, num)]
    for j in range(0, num):
        u[j] = 1 / num * j
        polynomials = [0 for k in range(0, num)]
    for i in range(0,num):
        polynomials[i] = [2*u[i]**3-3*u[i]**2+1, -2*u[i]**3+3*u[i]**2, u[i]**3-2*u[i]**2+u[i], u[i]**3-u[i]**2]
    points = [0 for i in range(0, num)]  # initialize points vector
    for i in range(0, num):
        points[i] = np.matmul(polynomials[i][:], np.transpose(cp))  # calculate each point value
    return points

#Function 5: cubic hermite surface computation
def HermiteSurface(cpp,num_u,num_v):
    u = [0 for j in range(0, num_u)]
    v = [0 for j in range(0, num_v)]
    for j in range(0, num_u):
        u[j] = 1 / num_u * j
    for j in range(0, num_v):
        v[j] = 1 / num_v * j
    u_poly = [0 for k in range(0, num_u)]
    v_poly = [0 for k in range(0, num_v)]
    for i in range(0, num_u):
        u_poly[i] = [u[i]**3, u[i]**2, u[i], 1]
    for i in range(0, num_v):
        v_poly[i] = [v[i]**3, v[i]**2, v[i], 1]
    x = cpp[:][:][0]
    y = cpp[:][:][1]
    z = cpp[:][:][2]
    X = np.zeros((num_u, num_v))  # initialize data points
    Y = np.zeros((num_u, num_v))
    Z = np.zeros((num_u, num_v))
    for i in range(num_u):
        for j in range(num_v):
            ux = np.matmul(u_poly[i][:], x)  # multiply U*point
            uy = np.matmul(u_poly[i][:], y)
            uz = np.matmul(u_poly[i][:], z)
            X[i][j] = np.matmul(ux, v_poly[:][j])  # multiply U*point*V
            Y[i][j] = np.matmul(uy, v_poly[:][j])
            Z[i][j] = np.matmul(uz, v_poly[:][j])
    points = [X, Y, Z]  # return points in 3xUxV sized array
    return points#

#Function 6: draw figures on GUI
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

#Function 7: draw initial chart
def drawChart():
    _VARS['pltFig'] = plt.figure()
    plt.plot(0, 0)
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

# Function 8: Update plot
def updateChart(data):
    plt.clf()
    data = np.array(data)
    if np.shape(data)[1] == 3:
        ax = plt.axes(projection='3d')
        ax.plot3D(data[:, 0], data[:, 1], data[:, 2])
    elif np.shape(data)[1] == 2:
        plt.plot(data[:, 0], data[:, 1])
    else:
        ax = plt.axes(projection='3d')
        ax.plot_surface(points[0][:][:], points[1][:][:], points[2][:][:])
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

#Function 9: delete previous chart
def deleteChart():
    _VARS['fig_agg'].get_tk_widget().forget()
    return

#Function 10: Process Data
def processData(values):
    data = values[0]
    data = data.replace('(', "")
    data = data.replace(')', "")
    data = data.replace('\n', " ")
    data = np.array([x.strip().split(',') for x in data.split(' ')], dtype=float)
    return data

#Function 11: reorganize point matrix into 3xn
def organizeMatrix(data):
        data = np.array(data)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        matrix = [[np.reshape(x, (4, 4))], [np.reshape(y, (4, 4))], [np.reshape(z, (4, 4))]]
        return matrix

#Function 12: Process rotation and translation values
def processTransform(values):
    rotation = values[1]
    translation = values[2]

    rotation = rotation.replace('(', "")
    rotation = rotation.replace(')', "")
    rotation = rotation.replace('\n', " ")
    rotation = np.array([x.strip().split(',') for x in rotation.split(' ')], dtype=float)

    translation = translation.replace('(', "")
    translation = translation.replace(')', "")
    translation = translation.replace('\n', " ")
    translation = np.array([x.strip().split(',') for x in translation.split(' ')], dtype=float)


    return rotation, translation

#Function 13: Process 3d rotation matrices
def processRotation(rotation):
    x = rotation[0][0]
    y = rotation[0][1]
    z = rotation[0][2]
    Rx = np.array([[1, 0, 0],
               [0, np.cos(x), -np.sin(x)],
               [0, np.sin(x), np.cos(x)]])

    Ry = np.array([[np.cos(y), 0, np.sin(y)],
               [0, 1, 0],
               [-np.sin(y), 0, np.cos(y)]])

    Rz = np.array([[np.cos(z), -np.sin(z), 0],
               [np.sin(z), np.cos(z), 0],
               [0, 0, 1]])
    R = np.dot(np.dot(Rx, Ry), Rz)
    return R

#Function 14: Apply rotation and transformations to curves
def Transform(points,rotation,translation):
    points = np.array(points)
    translation = np.array(translation)
    rotation = np.array(rotation)
    rotation = np.deg2rad(rotation)
    if np.shape(points)[1] == 2:
        #apply 2D translation
        points[:, :] = points[:, :] + translation
        #apply 2D rotation
        c = np.cos(rotation[0][0])
        s = np.sin(rotation[0][0])
        R = np.array([[c, -s], [s, c]])
        pos = [0 for j in range(0, np.shape(points)[0])]
        for i in range(0, np.shape(points)[0]):
            xyz = np.matmul(R, np.transpose(points[i]))
            pos[i] = xyz
        points = pos
    elif np.shape(points)[1] == 3:
        #Apply 3D translation
        points[:][:] = points[:][:] + translation
        #Apply 3D rotation
        rotation_matrix = processRotation(rotation)
        pos = [0 for j in range(0, np.shape(points)[0])]
        print(np.shape(pos))
        for i in range(0, np.shape(points)[0]):
            xyz = np.matmul(rotation_matrix, (points[i]))
            pos[i] = xyz
        points = pos
    else:
        #Apply 3D translation
        points[0][:][:] = points[0][:][:] + translation[0][0]
        points[1][:][:] = points[1][:][:] + translation[0][1]
        points[2][:][:] = points[2][:][:] + translation[0][2]
        #Apply 3D rotation
        R = processRotation(rotation)
        pos = np.array(points)
        for i in range(0, num_u):
            for j in range(0, num_v):
                x = points[0][i][j]
                y = points[1][i][j]
                z = points[2][i][j]
                xyz = np.matmul(R, [x, y, z])
                pos[0][i][j] = xyz[0]
                pos[1][i][j] = xyz[1]
                pos[2][i][j] = xyz[2]
        points = pos
        print(pos)
    return points

#Function 15: Scale
def Scale(points, values):
    scale = values[3]
    scale = scale.replace('(', "")
    scale = scale.replace(')', "")
    scale = scale.replace('\n', " ")
    scale = np.array([(x.strip().split(',')) for x in scale.split(' ')], dtype=float)
    points = np.array(points)
    if np.shape(data)[1] == 3:
        x = scale[0][0]
        y = scale[0][1]
        z = scale[0][2]
        points[:, 0] = points[:, 0]*x
        points[:, 1] = points[:, 1]*y
        points[:, 2] = points[:, 2]*z
    elif np.shape(data)[1] == 2:
        x = scale[0][0]
        y = scale[0][1]
        points[:, 0] = points[:, 0]*x
        points[:, 1] = points[:, 1]*y
    else:
        x = scale[0][0]
        y = scale[0][1]
        z = scale[0][2]
        points[0][:][:] = points[0][:][:] * x
        points[1][:][:] = points[1][:][:] * y
        points[2][:][:] = points[2][:][:] * z
    return points

# Global Variables for plotting
_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False}


num = 100   # initialize number of data points to approximate with
num_u = 100
num_v = 100
## GUI creation for visual aid
layout = [
    [sg.Text("Surface and Curve Visualizer", pad=((0, 250), (0, 0)), text_color='Black', background_color='#FDF6E3')],
    [sg.Canvas(key='figCanvas', background_color='#FDF6E3')],
    [sg.Text('Input x,y,z coordinates:', text_color='Black', background_color='#FDF6E3'), sg.Multiline(size=(30, 5), pad=((0, 150), (0, 25)))],
    [sg.Button('Hermite Curve'), sg.Button('Cubic Hermite Surface'),
     sg.Button('Bezier Curve'),  sg.Button('Cubic Bezier Surface'),
     sg.Button('Plot point-to-point'), sg.Button('Clear Graph'), ],
    [sg.Text('Insert rotation (x,y,z) in degrees', text_color='Black', background_color='#FDF6E3'), sg.InputText(size=(5, 5)),
     sg.Text('Insert Translation (x,y,z)', text_color='Black', background_color='#FDF6E3'), sg.InputText(size=(5, 5)),
     sg.Button('Apply Transform')],
    [sg.Text('Scale plot (x,y,z)', text_color='Black', background_color='#FDF6E3'), sg.InputText(size=(5, 5)),
     sg.Button('Apply Scale'),
     sg.Button('Exit')],

]

#GUI Themes
sg.theme('black')
plt.style.use('Solarize_Light2')

_VARS['window'] = sg.Window('ME6104',
                            layout,
                            finalize=True,
                            resizable=True,
                            element_justification="right",
                            background_color='#FDF6E3')

## GUI
drawChart()

# MAIN LOOP
while True:

    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Plot point-to-point':
        try:
            data = processData(values)
            deleteChart()
            points = data
            updateChart(points)
        except: 'ValueError'
    elif event == 'Bezier Curve':
     try:
        data = processData(values)
        points = BezierCurve(np.transpose(data), num)
        deleteChart()
        updateChart(points)
     except: 'ValueError'
    elif event == 'Hermite Curve':
        try:
            data = processData(values)
            points = HermiteCurve(np.transpose(data), num)
            deleteChart()
            updateChart(points)
        except:'ValueError'
    elif event == 'Clear Graph':
        deleteChart()
    elif event == 'Cubic Hermite Surface':
        try:
            data = processData(values)
            data = organizeMatrix(data)
            points = HermiteSurface(data, num_u, num_v)
            deleteChart()
            updateChart(points)
        except: 'ValueError'
    elif event == 'Cubic Bezier Surface':
        try:
            data = processData(values)
            data = organizeMatrix(data)
            points = BezierSurface(data, num_u, num_v)
            deleteChart()
            updateChart(points)
        except: 'ValueError'
    elif event == 'Apply Transform':
         #try:
            [rotation, translation] = processTransform(values)
            points = Transform(points, rotation, translation)
            deleteChart()
            updateChart(points)
         #except: 'ValueError'
    elif event == 'Apply Scale':
        try:
            points = Scale(points, values)
            deleteChart()
            updateChart(points)
        except: 'ValueError'




_VARS['window'].close()
