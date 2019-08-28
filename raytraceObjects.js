function BaseObject(typeName, implementsObject){
	var result = function(args){
		if(this instanceof arguments.callee){
			if(typeof this.init === 'function'){
				this.init.apply(this, args != undefined ? (args.callee ? args : arguments) : null);
			}
		}else {
			return new arguments.callee(arguments);
		}
	}

	if(implementsObject){
		result.prototype = Object.create(implementsObject.prototype);
	}

	result.prototype.getType = function(){ return typeName };

	return result;
}

var Light = BaseObject("Light", null);
{
	Light.prototype.init = function(r, g, b){
		this.colour = vec3(r, g, b);
		this.enabled = true;
	}

	Light.prototype.getLightVector = function(inPoint, outVector){
		if(!outVector){
			outVector = new vec3();
		}

		return outVector;
	}
}

var DirectionalLight = BaseObject("DirectionalLight", Light);
{
	Light.prototype.init = function(r, g, b, x, y, z){
		Light.prototype.init.call(this, r, g, b);
		this.direction = vec3(x, y, z);

		vec3.normalize(this.direction, this.direction);
	}

	Light.prototype.getLightVector = function(inPoint, outVector){
		var o = Light.prototype.getLightVector.call(this, inPoint, outVector);

		o[0] = this.direction[0];
		o[1] = this.direction[1];
		o[2] = this.direction[2];

		return o;
	}
}

var Material = BaseObject("Material", null);
{
	Material.prototype.init = function(){
		this.reflective = 0.0;
		this.refractive = 0.0;
		this.opacity = 1.0;
		this.diffuseColour = new vec3(1.0, 1.0, 1.0);
		this.specColour = new vec3(1.0, 1.0, 1.0);
	}

	Material.prototype.caclulateOutput = function(eyeVector, normalVector, point, light, outReflective, outRefractive, outColour){
		if(!outColour){
			outColour = new vec3();
		}

		outColour[0] = this.diffuseColour[0];
		outColour[1] = this.diffuseColour[1];
		outColour[2] = this.diffuseColour[2];

		return outColour;
	}
}

var RaytraceableObject = BaseObject("RaytraceableObject", null);
{
	RaytraceableObject.prototype.init = function(material, x, y, z){
		this.material = material;
		this.mvMatrix = new mat4();

		mat4.identity(this.mvMatrix);
		mat4.translate(x, y, z, this.mvMatrix);
	}

	/**
	* Obtains the lowest point grid aligned bounds for the object. The lowest point is defined as the leastmost x, y, and z components.
	*/
	RaytraceableObject.prototype.getLowBounds = function(){
		return this.location;
	}

	/**
	* Obtains the highest point grid aligned bounds for the object. The heighest point is defined as the most x, y, and z components.
	*/
	RaytraceableObject.prototype.getHighBounds = function(){
		return this.location;
	}

	var tPointBuffer = new vec3();
	RaytraceableObject.prototype.getSignedDistanceToWorldPoint = function(point){
		//TODO: transform point
		var tp = vec3.multVec3(this.mvMatrix, point, tPointBuffer);

		return this.getSignedDistanceToPoint(tp);
	}

	RaytraceableObject.prototype.getSignedDistanceToPoint = function(point){
		return Number.MAX_VALUE;
	}

	/**
	* Checks if the line intersects with the vector from a position.
	* returns false if no intersection occurs. If one does occur, then the intersection point is sent to the outPoint value.
	*/
	RaytraceableObject.prototype.doesIntersect = function(vector, origin, outPoint){

	}
}

var Sphere = BaseObject("Sphere", RaytraceableObject);
{
	Sphere.prototype.init = function(material, x, y, z){
		
	}
}