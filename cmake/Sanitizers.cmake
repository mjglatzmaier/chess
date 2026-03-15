function(havoc_enable_sanitizers target)
    if(NOT HAVOC_ENABLE_SANITIZERS)
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
            $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
            $<$<CONFIG:Debug>:-fno-sanitize-recover=all>
        )
        target_link_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        )
    elseif(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:/fsanitize=address>
        )
    endif()
endfunction()
