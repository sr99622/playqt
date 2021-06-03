#ifndef CUDAEXCEPTION_H
#define CUDAEXCEPTION_H

#include <QString>
#include <npp.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <exception>
#include <stdexcept>

using namespace std;

/*
enum cudaFamily {
    Cuda,
    Npp,
};

class CudaException : public runtime_error
{
public:
    CudaException(const string& arg);

    virtual const char* what() const noexcept {
        return runtime_error::what();
    }

};
*/

class CudaExceptionHandler
{
public:
    CudaExceptionHandler();

    void ck(const cudaError& ce, const QString& note = "");
    void ck(const NppStatus& status, const QString& note = "");

    static QString getErrorText(const NppStatus& status);

};

#endif // CUDAEXCEPTION_H
