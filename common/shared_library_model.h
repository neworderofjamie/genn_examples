#pragma once

// Standard C++ includes
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
// POSIX C includes
extern "C"
{
#include <dlfcn.h>
}
#endif

//----------------------------------------------------------------------------
// SharedLibraryModel
//----------------------------------------------------------------------------
template<typename scalar = float>
class SharedLibraryModel
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef void (*VoidFunction)(void);


    SharedLibraryModel() : m_Library(nullptr), m_AllocateMem(nullptr),
        m_Initialize(nullptr), m_InitializeSparse(nullptr),
        m_StepTimeGPU(nullptr), m_StepTimeCPU(nullptr)
    {
    }

    SharedLibraryModel(const std::string &modelName)
    {
        if(!open(modelName)) {
            throw std::runtime_error("Unable to open library");
        }
    }

    ~SharedLibraryModel()
    {
        // Close model library if loaded successfully
        if(m_Library)
        {
#ifdef _WIN32
            FreeLibrary(m_Library);
#else
            dlclose(m_Library);
#endif
        }
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool open(const std::string &modelName)
    {
#ifdef _WIN32
        const std::string libraryName = modelName + "_CODE\\runner.dll";
        m_Library = LoadLibrary(libraryName.c_str());
#else
        const std::string libraryName = modelName + "_CODE/librunner.so";
        m_Library = dlopen(libraryName.c_str(), RTLD_NOW);
#endif

        // If it fails throw
        if(m_Library != nullptr) {
            m_AllocateMem = (VoidFunction)getSymbol("allocateMem");
            m_Initialize = (VoidFunction)getSymbol("initialize");
            m_InitializeSparse = (VoidFunction)getSymbol("init" + modelName);

            m_StepTimeCPU = (VoidFunction)getSymbol("stepTimeCPU", true);
            m_StepTimeGPU = (VoidFunction)getSymbol("stepTimeGPU", true);

            m_T = (scalar*)getSymbol("t");
            return true;
        }
        else {
#ifdef _WIN32
            std::cerr << "Unable to load library - error:" << std::to_string(GetLastError()) << std::endl;;
#else
            std::cerr << "Unable to load library - error:" << dlerror() << std::endl;
#endif
            return false;
        }

    }

    void *getSymbol(const std::string &symbolName, bool allowMissing = false)
    {
#ifdef _WIN32
        void *symbol = GetProcAddress(m_Library, symbolName.c_str());
#else
        void *symbol = dlsym(m_Library, symbolName.c_str());
#endif

        // If this symbol isn't allowed to be missing but it is, raise exception
        if(!allowMissing && symbol == nullptr) {
            throw std::runtime_error("Cannot find symbol '" + symbolName + "'");
        }

        return symbol;
    }

    void allocateMem()
    {
        m_AllocateMem();
    }

    void initialize()
    {
        m_Initialize();
    }

    void initializeSparse()
    {
        m_InitializeSparse();
    }

    void stepTimeGPU()
    {
        m_StepTimeGPU();
    }

    void stepTimeCPU()
    {
        m_StepTimeCPU();
    }

    scalar getT() const
    {
        return *m_T;
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
#ifdef _WIN32
    HMODULE m_Library;
#else
    void *m_Library;
#endif

    VoidFunction m_AllocateMem;
    VoidFunction m_Initialize;
    VoidFunction m_InitializeSparse;
    VoidFunction m_StepTimeGPU;
    VoidFunction m_StepTimeCPU;

    scalar *m_T;
};

//----------------------------------------------------------------------------
// Typedefines
//----------------------------------------------------------------------------
typedef SharedLibraryModel<float> SharedLibraryModelFloat;
typedef SharedLibraryModel<double> SharedLibraryModelDouble;