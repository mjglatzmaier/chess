function(havoc_set_warnings target)
    set(GCC_CLANG_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wcast-align
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wformat=2
        -Wimplicit-fallthrough
        -Wmisleading-indentation
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wdouble-promotion
    )

    # Some flags are GCC-only; filter for Clang
    set(CLANG_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wcast-align
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wformat=2
        -Wimplicit-fallthrough
        -Wdouble-promotion
    )

    set(MSVC_WARNINGS
        /W4
        /permissive-
        /w14242  # conversion, possible loss of data
        /w14254  # operator conversion, possible loss of data
        /w14263  # function does not override base class virtual
        /w14265  # class has virtual functions but destructor is not virtual
        /w14287  # unsigned/negative constant mismatch
        /w14296  # expression is always true/false
        /w14311  # pointer truncation
        /w14545  # ill-formed comma expression
        /w14546  # function call before comma missing argument list
        /w14547  # operator before comma has no effect
        /w14549  # operator before comma has no effect
        /w14555  # expression has no effect
        /w14619  # pragma warning: nonexistent warning number
        /w14640  # thread-unsafe static member initialization
        /w14826  # conversion is sign-extended
        /w14905  # wide string literal cast to LPSTR
        /w14906  # string literal cast to LPWSTR
        /w14928  # illegal copy-initialization
    )

    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:${GCC_CLANG_WARNINGS}>
        $<$<CXX_COMPILER_ID:Clang>:${CLANG_WARNINGS}>
        $<$<CXX_COMPILER_ID:AppleClang>:${CLANG_WARNINGS}>
        $<$<CXX_COMPILER_ID:MSVC>:${MSVC_WARNINGS}>
    )
endfunction()
