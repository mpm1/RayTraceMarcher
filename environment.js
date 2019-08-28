

/**
A model that splits on a plane and add all elements that are less than and greater than the plane.
Split order: x plane, y plane, z plane, repeat
*/
var SplitTree3D = BaseObject("SplitTree3D", null);
{
	SplitTree3D.prototype.init = function(centreX, centreY, centreZ, splitOrder = 0){
		this.left = null;
		this.right = null;
		this.spliptOrder = splitOrder;
		this.centre = new vec3(centreX, centreY, centreZ);
		this.objects = null;
	}

	SplitTree3D.prototype.addChild = function(child){
		if(objects == null){
			this.objects = child;
		}else if(this.left == null){
			
		}
	}
}

var OctTree = BaseObject("OctTree", null);
{
	function addObject(child){
		if(this.objects){
			return false;
		}

		this.objects = child;

		return true;
	}

	/**
	* Octants
	* 0 = -1, -1, -1
	* 1 = -1, -1, 1
	* 2 = -1, 1, -1
	* 3 = -1, 1, 1
	* 4 = 1, -1, -1
	* 5 = 1, -1, 1
	* 6 = 1, 1, -1
	* 7 = 1, 1, 1
	*/
	OctTree.prototype.init = function(centreX, centreY, centreZ){
		this.children = new Array(8);
		this.centre = new vec3(centreX, centreY, centreZ);
		this.objects = null;
	}

	OctTree.prototype.addChild = function(child){
		if(this.hasChildren()){

		}else if(!addObject.call(this, child)){

		}
	}

	OctTree.prototype.hasChildren = function(){
		if(!(children[0])){
			return false;
		}

		return true;
	}
}

var Environment = BaseObject("Environment", null);
{
	Environment.prototype.init = function(){
		this.objects = [];
		this.lights = [];
		this.materials = [];
	}

	Environment.prototype.addObject = function(obj){
		this.objects.push(obj);
	}

	Environment.prototype.removeObject = function(obj){

	}

	Environment.prototype.addLight = function(light){
		this.lights.push(light);
	}

	Environment.prototype.removeLight = function(light){

	}

	var tTotal = 0;
	Environment.prototype.update = function(time) {
		tTotal += time;

		if(tTotal > 1000){
			++index;
			if(index > 4){
				index = 0;
			}

			tTotal = 0;
		}
	}

	var index = 0;
	var rLoop = [0.0, 0.25, 0.5, 0.75, 1.0];
	var gLoop = [0.0, 0.0625, 0.25, 0.5625, 1.0];
	var bLoop = [0.0, 0.015625, 0.125, 0.421875, 1.0];
	Environment.prototype.draw = function(ray, point, influenceRad, eyeLocation, outColour){
		outColour[0] = rLoop[index];
		outColour[1] = gLoop[index];
		outColour[2] = bLoop[index];
	}
}

var Camera = BaseObject("Camera", null);
{
	Camera.prototype.init = function(x, y, z, rx, ry, rz, viewRatio, angle){
		this.location = new vec3(x, y, z);
		this.rotation = new vec3(rx, ry, rz);
		this.mvMatrix = new mat4();
		this.viewRatio = viewRatio;
		this.angle = angle;
		this.frontPane = 5.0; // TODO calculate by angle

		this.update(0);
	}

	Camera.prototype.update = function(time){
		var m = this.mvMatrix;
		var l = this.location;
		mat4.identity(this.mvMatrix);


		mat4.translate(l[0], l[1], l[2], m);
		mat4.rotateY(this.rotation[1], m);
	}

	var speed = 0.015;
	Camera.prototype.move = function(time, moveVec){
		var l = this.location;
		var d = time * speed;
		var x = moveVec[0] * d;
		var y = moveVec[1] * d;
		var z = moveVec[2] * d;
		var s = Math.sin(-this.rotation[1]);
		var c = Math.cos(-this.rotation[1]);

		l[0] += (x * c) - (z * s);
		l[1] += y;
		l[2] += (x * s) + (z * c);
	}

	var twoPi = Math.PI * 2.0;
	Camera.prototype.rotateY = function(radians){
		this.rotation[1] += radians;

		while(this.rotation[1] > twoPi){
			this.rotation[1] -= twoPi;
		}

		while(this.rotation[1] < -twoPi){
			this.rotation[1] += twoPi;
		}
	}

	Camera.prototype.getRay = function(u, v, outRay){
		outRay[2] = this.frontPane;
		outRay[1] = (v - 0.5) * 0.5;
		outRay[0] = (u - 0.5) * 0.5;

		outRay = vec3.normalize(outRay, outRay);

		return outRay;
	}
}

var Light = BaseObject("Light", null);
{
	Light.prototype.init = function(r, g, b, x, y, z, softFactor, castShadow, power, type){
		this.r = r;
		this.g = g;
		this.b = b;
		this.x = x;
		this.y = y;
		this.z = z;
		this.softFactor = softFactor;
		this.castShadow = castShadow;
		this.objType = type;
		this.power = power;
	}
}

var DirectionalLight = BaseObject("DirectionalLight", Light);
{
	DirectionalLight.prototype.init = function(r, g, b, x, y, z, softFactor, castShadow, power){
		Light.prototype.init.call(this, r, g, b, x, y, z, softFactor, castShadow, power, 0);
	}
}

var PointLight = BaseObject("PointLight", Light);
{
	PointLight.prototype.init = function(r, g, b, x, y, z, softFactor, castShadow, power){
		Light.prototype.init.call(this, r, g, b, x, y, z, softFactor, castShadow, power, 1);
	}
}

var WorldRenderer = BaseObject("Renderer", null);
{
	WorldRenderer.prototype.init = function(canvas, environmet){
		this.canvas = canvas;
		this.context = canvas.getContext("2d");
		this.imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
		this.environmet = environmet;
		this.camera = new Camera(0, 0, 0, 0, 0, 0, canvas.width/canvas.height, 45.0);
	}

	WorldRenderer.prototype.update = function(timeDelta){
		this.environmet.update(timeDelta);
		this.camera.update(timeDelta);
	}

	var ray = new vec3();
	var colour = new vec3();
	WorldRenderer.prototype.draw = function(offset, increment){
		// With these properties we can draw using checkerboard rendering
		var data = this.imageData.data;
		var total = data.length;
		var mulIncr = increment << 2;

		var x = offset % this.canvas.width;
		var y = Math.floor(offset / this.canvas.width);

		for(var i = offset << 2; i < total; i += mulIncr){
			this.camera.getRay(x / this.canvas.width, y / this.canvas.height, ray);
			this.environmet.draw(ray, this.camera.location, colour);

			x += increment;
			if(x >= this.canvas.width){
				x -= this.canvas.width;
				++y;
			}

			data[i] = Math.floor(colour[0] * 255);
			data[i + 1] = Math.floor(colour[1] * 255);
			data[i + 2] = Math.floor(colour[2] * 255);
			data[i + 3] = 255;
		}

		this.context.putImageData(this.imageData, 0, 0);
	}
}

var Texture = BaseObject("Texture", null);
{
	var lastId;

	Texture.prototype.init = function(imageUrl){
		this.image = new Image();
		this.glImage = null;
		
		var isLoaded = false;

		this.image.onload = function(){
			isLoaded = true;
		}

		this.image.src = imageUrl;

		this.isLoaded = function(){
			return isLoaded;
		}
	}
}

var Material = BaseObject("Material", null);
{
	var lastId = 0;

	Material.prototype.init = function(diffuseColour, specularColour){
		this.id = ++lastId;
		this.diffuseColour = vec3(diffuseColour[0], diffuseColour[1], diffuseColour[2]);
		this.specularColour = vec3(specularColour[0], specularColour[1], specularColour[2]);
		this.roughness = 0.0;
		this.reflectionAmount = 0.0;
		this.diffuseTexture = null;
		this.normalTexture = null;
		this.roughnessTexture = null;
	}
}

var Shape = BaseObject("Shape", null);
{
	var lastId = 0;

	Shape.prototype.init = function(location, radius, materialId, type){
		this.id = ++lastId;
		this.location = location;
		this.radius = radius;
		this.materialId = materialId;
		this.type = type;
	}
}

var Sphere = BaseObject("Sphere", Shape);
{
	Sphere.prototype.init = function(location, radius, materialId){
		Shape.prototype.init.call(this, location, radius, materialId, 0);
	}
}

var Plane = BaseObject("Plane", Shape);
{
	Plane.prototype.init = function(location, height, materialId){
		Shape.prototype.init.call(this, location, height, materialId, 1);
	}
}

var Cube = BaseObject("Cube", Shape);
{
	Cube.prototype.init = function(location, radius, materialId){
		Shape.prototype.init.call(this, location, radius, materialId, 2);
	}
}

var Model = BaseObject("Model", Shape);
{
	Model.prototype.init = function(locaiton, radius, materialId, modelJson){
		Shape.prototype.init.call(this, location, radius, materialId, 2);
	}
}