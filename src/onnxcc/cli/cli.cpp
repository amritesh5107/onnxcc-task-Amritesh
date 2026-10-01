#include"cli.h"
#include<cxxopts.hpp>
#include<iostream>
#include<string>

namespace onnxcc::cli{

    int run_dump(int argc,char* argv[]){
        // object of class Options with constructor calls
        cxxopts::Options options("onnxcc","ONNX compiler"); 

        options.add_options()
            ("m,model","Path to ONNX model",cxxopts::value<std::string>())
            ("show-graph","Show the model graph")
            ("verbose","Enable verbose output")
            ("h,help","Show help");

        cxxopts::ParseResult result; //parses and stores in parseresult.

        //argv[0] is the first element so we point one step ahead and pnly take argc-1 elements.
        //try catch loop which catches invalid commands 
        try{
            result=options.parse(argc-1,argv+1);
        }
        catch(const cxxopts::exceptions::exception& e){
            std::cerr<<"Error: "<<e.what()<<"\n";
            return 1;
        }   

        if(result.count("help")){
            std::cout<<options.help()<<"\n";
            return 0;
        }

        if(!result.count("model")){
            std::cerr<<"Model is not given\n";
            return 1;
        }
        const std::string model_path=result["model"].as<std::string>(); 
        //const is being used because we do not need to change model path....We only read it.

        const bool show_graph=result.count("show-graph")>0;
        const bool verbose=result.count("verbose")>0;

        std::cout<<"model "<<model_path<<"\n";
        std::cout<<"show-graph "<<(show_graph ? "yes":"no")<<"\n";
        std::cout<<"verbose "<<(verbose ? "yes":"no")<<"\n";

        return 0;
    }
    int run(int argc,char* argv[]){
        if(argc<2){
            std::cerr<<"Error:No command specified\n";
            return 1;
        }
        std::string command_one=std::string(argv[1]);

        if (command_one=="dump") {
            return run_dump(argc,argv);
        }

        std::cerr<<"Error:Invalid command: "<<command_one<<"\n";
        return 1;
    }
}