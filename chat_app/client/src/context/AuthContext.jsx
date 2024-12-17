import { createContext, useCallback, useState } from "react";


export const AuthContextProvider = ({children}) => {
    const [user, setUser] = useState(null);
    const [registerInfo, setRegisterInfo] = useState({
        name:"",
        email:"",
        password:"",
    });

    const updateRegisterInfo = useCallback((info) => {
        setRegisterInfo(info);
    }, [])

    console.log("register info", registerInfo)

    return <AuthContext.Provider 
        value = {{
            user,
            registerInfo,
            updateRegisterInfo
        }}
    >
        {children}
    </AuthContext.Provider>
}

export const AuthContext = createContext()
