#include <gtest/gtest.h>
#include "onnxcc/version.h"
#include "onnxcc/cli/cli.h"
#include <string>

// Trivial test to ensure gtest is working
TEST(SanityCheck, GTESTWorks) {
    EXPECT_TRUE(true);
}

// Trivial test to ensure version is not empty
TEST(SanityCheck, VersionCheck) {
    std::string version = std::string(onnxcc::get_version());
    EXPECT_FALSE(version.empty());
}

// The parameters passed to check are basically test group and test description

// Test for checking command.
TEST(Cli_test,Dump_with_model){
    const char* argv[]={  
        "onnxcc","dump","--model","test.onnx"  //If we do not use const then cpp gives warning
    };

    int result=onnxcc::cli::run(4,const_cast<char**>(argv)); 
    EXPECT_EQ(result,0); //expected return val shd be 0 means successful
}

TEST(Cli_test,Dump_without_model){
    //Each argv[i] is a pointer to const char
    const char* argv[]={
        "onnxcc","dump"
    };
    int result=onnxcc::cli::run(2,const_cast<char**>(argv));
    EXPECT_NE(result,0); //Expected to fail
}

TEST(Cli_test,Dump_with_invalid_command){
    const char* argv[]={ 
        "onnxcc","dump","--vdwwc"
    };
    int result=onnxcc::cli::run(3,const_cast<char**>(argv));
    EXPECT_NE(result,0);
}

TEST(Cli_test,Dump_help){
    const char* argv[]={
        "onnxcc","dump","--help"
    };
    int result=onnxcc::cli::run(3,const_cast<char**>(argv));
    EXPECT_EQ(result,0);
}