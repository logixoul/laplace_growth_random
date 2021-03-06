#include "precompiled.h"
#include "shade.h"
#include "util.h"
#include "qdebug.h"
#include "stuff.h"

void beginRTT(gl::TextureRef fbotex)
{
	static unsigned int fboid = 0;
	if(fboid == 0)
	{
		glGenFramebuffersEXT(1, &fboid);
	}
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboid);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fbotex->getId(), 0);
}
void endRTT()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

static void drawRect() {
	auto ctx = gl::context();

	ctx->getDrawTextureVao()->bind();
	//ctx->getDrawTextureVbo()->bind(); // this seems to be unnecessary

	ctx->drawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

std::map<string, float> globaldict;
void globaldict_default(string s, float f) {
	if(globaldict.find(s) == globaldict.end())
	{
		globaldict[s] = f;
	}
}

auto samplerSuffix = [&](int i) -> string {
	return (i == 0) ? "" : ToString(1 + i);
};

auto samplerName = [&](int i) -> string {
	return "tex" + samplerSuffix(i);
};

std::string getCompleteFshader(vector<gl::TextureRef> const& texv, std::string const& fshader) {
	auto texIndex = [&](gl::TextureRef t) {
		return ToString(
			1 + (std::find(texv.begin(), texv.end(), t) - texv.begin())
			);
	};
	string uniformDeclarations;
	FOR(i,0,texv.size()-1)
	{
		uniformDeclarations += "uniform sampler2D " + samplerName(i) + ";\n";
		uniformDeclarations += "uniform vec2 " + samplerName(i) + "Size;\n";
		uniformDeclarations += "uniform vec2 tsize" + samplerSuffix(i) + ";\n";
	}
	foreach(auto& p, globaldict)
	{
		uniformDeclarations += "uniform float " + p.first + ";\n";
	}
	string intro =
		Str()
		<< "#version 130"
		<< uniformDeclarations
		<< "vec3 _out = vec3(0.0);"
		<< "in vec2 tc;"
		<< "out vec4 OUTPUT;"
		<< "uniform vec2 mouse;"
		<< "vec4 fetch4(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).rgba;"
		<< "}"
		<< "vec3 fetch3(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).rgb;"
		<< "}"
		<< "vec2 fetch2(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).rg;"
		<< "}"
		<< "float fetch1(sampler2D tex_, vec2 tc_) {"
		<< "	return texture2D(tex_, tc_).r;"
		<< "}"
		<< "vec4 fetch4(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).rgba;"
		<< "}"
		<< "vec3 fetch3(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).rgb;"
		<< "}"
		<< "vec2 fetch2(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).rg;"
		<< "}"
		<< "float fetch1(sampler2D tex_) {"
		<< "	return texture2D(tex_, tc).r;"
		<< "}"
		<< "vec4 fetch4() {"
		<< "	return texture2D(tex, tc).rgba;"
		<< "}"
		<< "vec3 fetch3() {"
		<< "	return texture2D(tex, tc).rgb;"
		<< "}"
		<< "vec2 fetch2() {"
		<< "	return texture2D(tex, tc).rg;"
		<< "}"
		<< "float fetch1() {"
		<< "	return texture2D(tex, tc).r;"
		<< "}"
		<< "vec2 safeNormalized(vec2 v) { return length(v)==0.0 ? v : normalize(v); }"
		<< "vec3 safeNormalized(vec3 v) { return length(v)==0.0 ? v : normalize(v); }"
		<< "vec4 safeNormalized(vec4 v) { return length(v)==0.0 ? v : normalize(v); }";
	string outro =
		Str()
		<< "void main()"
		<< "{"
		<< "	OUTPUT.a = 1.0;"
		<< "	shade();"
		<< "	OUTPUT.rgb = _out;"
		<< "}";
	return intro + fshader + outro;
}

gl::TextureRef shade(vector<gl::TextureRef> const& texv, const char* fshader_constChar, ShadeOpts const& opts)
{
	const string fshader(fshader_constChar);

	static std::map<string, gl::GlslProgRef> shaders;
	gl::GlslProgRef shader;
	if(shaders.find(fshader) == shaders.end())
	{
		std::string completeFshader = getCompleteFshader(texv, fshader);
		try{
			auto fmt = gl::GlslProg::Format()
				.vertex(
					Str()
					<< "#version 150"
					<< "in vec4 ciPosition;"
					<< "in vec2 ciTexCoord0;"
					<< "out highp vec2 tc;"

					<< "void main()"
					<< "{"
					<< "	gl_Position = ciPosition * 2 - 1;"
					<< "	gl_Position.y = -gl_Position.y;"
					<< "	tc = vec2(0.0, 1.0) + vec2(1.0, -1.0) * ciTexCoord0;"
					<< "}")
				.fragment(completeFshader)
				.attribLocation("ciPosition", 0)
				.attribLocation("ciTexCoord0", 1)
				.preprocess(false);
			shader = gl::GlslProg::create(fmt);
			shaders[fshader] = shader;
		} catch(gl::GlslProgCompileExc const& e) {
			cout << "gl::GlslProgCompileExc: " << e.what() << endl;
			cout << "source:" << endl;
			cout << completeFshader << endl;
			throw;
		}
	} else {
		shader = shaders[fshader];
	}
	auto tex0 = texv[0];
	shader->bind();
	auto app=ci::app::App::get();
	float mouseX=app->getMousePos().x/float(app->getWindowWidth());
	float mouseY=app->getMousePos().y/float(app->getWindowHeight());
	shader->uniform("mouse", vec2(mouseX, mouseY));
	shader->uniform("tex", 0); tex0->bind(0);
	shader->uniform("texSize", vec2(tex0->getSize()));
	shader->uniform("tsize", vec2(1.0)/vec2(tex0->getSize()));
	foreach(auto& p, globaldict)
	{
		shader->uniform(p.first, p.second);
	}
	FOR(i, 1, texv.size()-1) {
		//shader.
		//string index = texIndex(texv[i]);
		shader->uniform(samplerName(i), i); texv[i]->bind(i);
		shader->uniform(samplerName(i) + "Size", vec2(texv[i]->getSize()));
		shader->uniform("tsize"+samplerSuffix(i), vec2(1)/vec2(texv[i]->getSize()));
	}
	gl::TextureRef result;
	ivec2 resultSize(tex0->getWidth() * opts._scaleX, tex0->getHeight() * opts._scaleY);
	GLenum ifmt = opts._ifmt.exists ? opts._ifmt.val : tex0->getInternalFormat();
	result = maketex(resultSize.x, resultSize.y, ifmt);

	beginRTT(result);
	
	gl::pushMatrices();
	{
		gl::ScopedViewport sv(ivec2(), result->getSize());
		gl::setMatricesWindow(ivec2(1, 1), true);
		::drawRect();
		gl::popMatrices();
	}

	endRTT();

	return result;
}
