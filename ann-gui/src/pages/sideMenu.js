import React, { useState } from 'react';
import { FormControl, FormLabel, Select, Button, Box, VStack } from "@chakra-ui/react";
import axios from 'axios';

const SideMenu = () => {
    // const [layers, setLayers] = useState([{ neurons: 2 }]);
    // const [activation, setActivation] = useState("logistic");
    // const [inputData, setInputData] = useState([0.5, 0.25]);
    // const [output, setOutput] = useState(null);

    // const handleForward = async () => {
    //     const response = await axios.post('http://localhost:5000/forward', {
    //       input: inputData,
    //       weights: [], 
    //       biases: [],   
    //       activation: activation
    //     });
    //     setOutput(response.data.output);
    //   };    

    return (
        <>
            {/* <FormControl>
                <FormLabel>Activation Function</FormLabel>
                <Select value={activation} onChange={(e) => setActivation(e.target.value)}>
                    <option value="logistic">Logistic</option>
                    <option value="relu">ReLU</option>
                    <option value="tanh">Tanh</option>
                </Select>
            </FormControl>

            <Button onClick={handleForward}>Forward Pass</Button>

            {output && <Box>Output: {output.join(', ')}</Box>} */}
            {/* Add a column of buttons */}
            <VStack spacing={2} padding={2}>
            <Button>Button 1</Button>
            <Button>Button 2</Button>
            <Button>Button 3</Button>
            <Button>Button 4</Button>
            </VStack>
        </>
    );
}

export default SideMenu;