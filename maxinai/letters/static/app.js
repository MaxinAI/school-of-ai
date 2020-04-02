(function () {

var size = 28
var width = Math.floor(Math.min(innerHeight, innerWidth)/(size + 2))

var imageData = []

for (var i = 0; i < size; i++) {
    imageData[i] = []
    for (var j = 0; j < size; j++) {
        imageData[i][j] = 0
    }
}

var canvas = document.getElementById('canvas')
canvas.width = canvas.height = size * width
canvas.style.cursor = 'crosshair'
canvas.style.width = canvas.style.height = size * width + 'px'

canvas.addEventListener('mousemove', mousemoveHandler, false)
canvas.addEventListener('mousedown', mousedownHandler, false)
canvas.addEventListener('mouseout', mouseoutHandler, false)
canvas.addEventListener('mouseover', mouseoverHandler, false)
document.addEventListener('mouseup', mouseupHandler, false)

var resetButton = document.getElementById('button-reset')
resetButton.addEventListener('click', function () {
    for (var i = 0; i < size; i++) {
        for (var j = 0; j < size; j++) {
            imageData[i][j] = 0
        }
    }
    clearCanvas()
    drawBackground()
    showResult()
    resultDiv.innerHTML = ''
}, false)

var resultButton = document.getElementById('button-result')
resultButton.addEventListener('click', function () {
    resultCanvas.toBlob(function (blob) {
        var xhr = new XMLHttpRequest()
        xhr.open('POST', 'WhatNumber', true)
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4) {
                var result = xhr.responseText
                if (isFinite(parseInt(result))) {
                	resultDiv.innerHTML = parseInt(result)
                } else {
                	resultDiv.innerHTML = '?'
                }
            }
        }
        xhr.send(blob)
    })
}, false)

var resultDiv = document.getElementById('div-result')
resultDiv.style.height = size * width + 'px'
resultDiv.style.fontSize = size * width + 'px'

var resultCanvas = document.getElementById('canvas-result')
resultCanvas.width = resultCanvas.height = size
resultCanvas.style.width = resultCanvas.style.height = size + 'px'
document.getElementById('span-dimension').innerHTML = size + ' x ' + size

var lineWidth = 1

document.getElementById('select').addEventListener('change', function () {
    lineWidth = this.value
})

var c = canvas.getContext('2d')

clearCanvas()
drawBackground()
showResult()

var mouseDown = false

function mousemoveHandler (e) {
	if (!mouseDown) return
	if (lastX == null || lastY == null) return
	
	var x = e.offsetX || e.clientX
	var y = e.offsetY || e.clientY

    draw(x, y)
	
	lastX = x
	lastY = y
}

function mouseupHandler (e) {
	mouseDown = false
}

function mousedownHandler (e) {
	mouseDown = true
	
	var x = e.offsetX || e.clientX
	var y = e.offsetY || e.clientY
	
	lastX = x
	lastY = y
	
	mousemoveHandler(e)
}

function mouseoverHandler (e) {
	var x = e.offsetX || e.clientX
	var y = e.offsetY || e.clientY
	
	lastX = x
	lastY = y
}

function mouseoutHandler (e) {
	mousemoveHandler(e)
}

function fillCell (i, j) {
    c.fillStyle = '#000'
    c.fillRect(i * width, j * width, width, width)
}

function repaint () {
    for (var i = 0; i < size; i++) {
        for (var j = 0; j < size; j++) {
            if (imageData[i][j]) {
                fillCell(i, j)
            }
        }
    }
}

function draw (x, y) {
    var i = Math.floor(x/width)
    var j = Math.floor(y/width)

    if (i >= size) i = size - 1
    if (j >= size) j = size - 1
    if (i < 0) i = 0
    if (j < 0) j = 0
    
    imageData[i][j] = 1
    
    for (var x = 1; x < lineWidth; x++) {
        (function (w) {
            if (j + w < size) {
                imageData[i][j + w] = 1
            }
            if (j - w >= 0) {
                imageData[i][j - w] = 1
            }
            
            if (i + w < size) {
                imageData[i + w][j] = 1
            }
            if (i - w >= 0) {
                imageData[i - w][j] = 1
            }
            
            if (j + w < size && i + w < size) {
                imageData[i + w][j + w] = 1
            }
            if (j + w < size && i - w >= 0) {
                imageData[i - w][j + w] = 1
            }
            
            if (j - w >= 0 && i + w < size) {
                imageData[i + w][j - w] = 1
            }
            if (j - w >= 0 && i - w >= 0) {
                imageData[i - w][j - w] = 1
            }
            
        })(x)
    }
    
    repaint()
    showResult()
}

function drawBackground () {
    c.strokeStyle = '#ccc'
    c.lineWidth = 1
    for (var i = 1; i < size; i++) {
        var p = i * width
        c.beginPath()
        c.moveTo(p, 0)
        c.lineTo(p, canvas.height)
        c.stroke()
        
        c.moveTo(0, p)
        c.lineTo(canvas.width, p)
        c.stroke()
    }
}

function clearCanvas () {
    c.fillStyle = '#fff'
    c.fillRect(0, 0, canvas.width, canvas.height)
}

function showResult () {
    var ctx = resultCanvas.getContext('2d')
    
    var imgData = ctx.getImageData(0, 0, size, size)
    var data = imgData.data
    
    for (var i = 0; i < size; i++) {
        for (var j = 0; j < size; j++) {
            var index = (j * size + i) * 4
            var color = imageData[i][j] ? 0 : 255
            
            data[index] = color
            data[index + 1] = color
            data[index + 2] = color
            data[index + 3] = 255
        }
    }
    
    ctx.putImageData(imgData, 0, 0)
}

})()
