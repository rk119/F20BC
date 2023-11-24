import React, { useState } from 'react';
import {
    Modal, ModalOverlay, ModalContent, ModalHeader, ModalFooter, ModalBody, ModalCloseButton,
    NumberInput, NumberInputField, FormControl, FormLabel, Select, Button, HStack, VStack, NumberIncrementStepper, NumberDecrementStepper, NumberInputStepper, Spacer
} from "@chakra-ui/react";

import axios from 'axios';

const SideMenu = ({ onBrowseData, setNeuralNetworkPlot, setAnnCreated, setAnnDetails, setResponseData }) => {
    const [isAnnModalOpen, setIsAnnModalOpen] = useState(false);
    const [isLayerConfigModalOpen, setIsLayerConfigModalOpen] = useState(false);
    const [numOfHidden, setHiddenLayers] = useState(2);
    const [localInputLayerNodes, setLocalInputLayerNodes] = useState(3);
    const [outputLayerNodes, setOutputLayerNodes] = useState(1); // Default value for output layer nodes
    const [layerConfigs, setLayerConfigs] = useState([]);
    const [lossFunction, setLossFunction] = useState('mse'); // Default loss function
    const [isHyperparametersModalOpen, setIsHyperparametersModalOpen] = useState(false);
    const [numOfParticles, setNumOfParticles] = useState(10);
    const [inertiaWeight, setInertiaWeight] = useState(0.5);
    const [cognitiveWeight, setCognitiveWeight] = useState(0.5);
    const [socialWeight, setSocialWeight] = useState(0.5);
    const [globalWeight, setGlobalWeight] = useState(0.5);
    const [numIterations, setNumIterations] = useState(100);
    const [numInformants, setNumInformants] = useState(3);
    const [inertiaWeights, setInertiaWeights] = useState({ min: 0, max: 1 });
    const [r1Range, setR1Range] = useState({ min: 0, max: 1 });
    const [r2Range, setR2Range] = useState({ min: 0, max: 1 });
    const [r3Range, setR3Range] = useState({ min: 0, max: 1 });

    const toggleAnnModal = () => {
        setIsAnnModalOpen(!isAnnModalOpen);
    };

    // Toggle function for hyperparameters modal
    const toggleHyperparametersModal = () => {
        setIsHyperparametersModalOpen(!isHyperparametersModalOpen);
    };

    const handleCreateAnnClick = () => {
        checkInputLayerNodes(); // Trigger the callback
        toggleAnnModal();       // Open the ANN modal
    };

    const handleAnnModalOk = () => {
        let initialConfigs = Array.from({ length: numOfHidden }, () => ({ neurons: 5, activation: 'relu' }));
        setLayerConfigs(initialConfigs);
        setIsAnnModalOpen(false);
        setIsLayerConfigModalOpen(true);
    };

    const handleNeuronChange = (index, value) => {
        const updatedConfigs = [...layerConfigs];
        updatedConfigs[index].neurons = value;
        setLayerConfigs(updatedConfigs);
    };

    const handleActivationChange = (index, value) => {
        const updatedConfigs = [...layerConfigs];
        updatedConfigs[index].activation = value;
        setLayerConfigs(updatedConfigs);
    };

    const handleCreateAnnOk = async () => {
        const annConfig = {
            inputSize: localInputLayerNodes,
            hiddenLayers: layerConfigs.map(layer => ({
                neurons: layer.neurons,
                activation: layer.activation
            })),
            outputLayer: {
                neurons: outputLayerNodes,
                activation: 'sigmoid' // Since it's default
            },
            lossFunction: lossFunction
        };
    
        const response = await axios.post('http://localhost:5000/createAnn', annConfig);
        console.log("ANN Creation Response:", response.data);

        setNeuralNetworkPlot(response.data.plot);
        setResponseData(null);
        setAnnCreated(true);
        setAnnDetails(response.data.ann_details);

        setIsLayerConfigModalOpen(false);
    };

    const handlePSOSubmit = async () => {
        // Construct the parameters object
        const psoParams = {
            no_solution: numOfParticles,
            iw_range: inertiaWeights,
            num_informants: numInformants,
            c: (cognitiveWeight, globalWeight, socialWeight, inertiaWeight),
            num_iterations: numIterations,
            r1_Range: r1Range,
            r2_Range: r2Range,
            r3_Range: r3Range,
        };
      
        // Send data to your API (modify this as per your API endpoint)
        const response = await axios.post('http://localhost:5000/runAnn', psoParams);
        // Handle the response...
      
        if (response.data && response.data.status === "success") {
          // Update the state with the received data
          // This can be used to display live updates in Window 1
          console.log("PSO Response:", response.data);
        }
      
        togglePSOModal(); // Close the modal
      };
      

    return (
        <>
            {/* Existing Buttons for Browse Data and Create ANN */}
            <VStack spacing={2} padding={2}>
                <Button onClick={onBrowseData} borderStyle={"solid"} borderColor={"black"} borderWidth={"3px"} _hover={{ bg: "teal.500", color: "white", }} width={150} height={50}>Browse Data</Button>
                <Button onClick={toggleAnnModal} borderStyle={"solid"} borderColor={"black"} borderWidth={"3px"} _hover={{ bg: "teal.500", color: "white", }} width={150} height={50}>Create ANN</Button>
                <Button onClick={toggleHyperparametersModal} borderStyle={"solid"} borderColor={"black"} borderWidth={"3px"} _hover={{ bg: "teal.500", color: "white", }} width={150} height={50}>Hyperparameters</Button>
                <Button borderStyle={"solid"} borderColor={"black"} borderWidth={"3px"} _hover={{ bg: "teal.500", color: "white", }} width={150} height={50}>Show Parameters</Button>
            </VStack>

            {/* First Modal for Number of Hidden Layers */}
            <Modal isOpen={isAnnModalOpen} onClose={toggleAnnModal}>
                <ModalOverlay />
                <ModalContent>
                    <ModalHeader>Create Artificial Neural Network</ModalHeader>
                    <ModalBody>
                        <FormControl id="hiddenLayers">
                            <FormLabel>Number of Hidden Layers</FormLabel>
                            <NumberInput size="lg" value={numOfHidden} onChange={(value) => setHiddenLayers(value)} focusBorderColor="teal.400" min={1} step={1}>
                                <NumberInputField />
                                <NumberInputStepper>
                                    <NumberIncrementStepper />
                                    <NumberDecrementStepper />
                                </NumberInputStepper>
                            </NumberInput>
                        </FormControl>
                    </ModalBody>
                    <ModalFooter>
                        <Button colorScheme="blue" mr={3} onClick={toggleAnnModal}>Cancel</Button>
                        <Button onClick={handleAnnModalOk}>OK</Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>

 
            {/* Second Modal for Layer Configuration */}
            <Modal isOpen={isLayerConfigModalOpen} onClose={() => setIsLayerConfigModalOpen(false)}>
                <ModalOverlay />
                <ModalContent>
                    <ModalHeader>Layer Configuration</ModalHeader>
                    <ModalBody>
                        <FormControl id="inputLayerNodes">
                            <FormLabel>Input Layer Neuron Count</FormLabel>
                            <NumberInput size="lg" value={localInputLayerNodes} onChange={(value) => setLocalInputLayerNodes(value)} focusBorderColor="teal.400" min={1} step={1}>
                                <NumberInputField />
                                <NumberInputStepper>
                                    <NumberIncrementStepper />
                                    <NumberDecrementStepper />
                                </NumberInputStepper>
                            </NumberInput>
                        </FormControl>
                        {layerConfigs.map((config, index) => (
                            <HStack key={index} mt={4}>
                                <FormControl id={`layer-${index}-neurons`}>
                                    <FormLabel>Hidden Layer {index + 1} Neuron #</FormLabel>
                                    <NumberInput size="lg" value={config.neurons} onChange={(value) => handleNeuronChange(index, value)} focusBorderColor="teal.400" min={1}>
                                        <NumberInputField />
                                        <NumberInputStepper>
                                            <NumberIncrementStepper />
                                            <NumberDecrementStepper />
                                        </NumberInputStepper>
                                    </NumberInput>
                                </FormControl>
                                <FormControl id={`layer-${index}-activation`}>
                                    <FormLabel>Activation Function</FormLabel>
                                    <Select value={config.activation} onChange={(e) => handleActivationChange(index, e.target.value)}>
                                        <option value="relu">ReLU</option>
                                        <option value="sigmoid">Sigmoid</option>
                                        <option value="tanh">Tanh</option>
                                    </Select>
                                </FormControl>
                            </HStack>
                        ))}
                         {/* Output Layer Configuration */}
                         <HStack mt={4}>
                            <FormControl id="outputLayerNodes">
                                <FormLabel>Output Layer Neuron #</FormLabel>
                                <NumberInput size="lg" value={outputLayerNodes} onChange={(value) => setOutputLayerNodes(value)} focusBorderColor="teal.400" min={1}>
                                    <NumberInputField />
                                    <NumberInputStepper>
                                        <NumberIncrementStepper />
                                        <NumberDecrementStepper />
                                    </NumberInputStepper>
                                </NumberInput>
                            </FormControl>

                            <FormControl id="outputLayerActivation">
                                <FormLabel>Activation Function</FormLabel>
                                <Select defaultValue="sigmoid">
                                    <option value="sigmoid">Sigmoid</option>
                                </Select>
                            </FormControl>
                        </HStack>

                        <FormControl mt={4}>
                            <FormLabel>Loss Function</FormLabel>
                            <Select value={lossFunction} onChange={(e) => setLossFunction(e.target.value)}>
                                <option value="mse">Mean Squared Error (MSE)</option>
                                <option value="binaryCrossEntropy">Binary Cross Entropy</option>
                                <option value="hinge">Hinge</option>
                            </Select>
                        </FormControl>
                    </ModalBody>
                    <ModalFooter>
                        <Button colorScheme="blue" mr={3} onClick={() => setIsLayerConfigModalOpen(false)}>Cancel</Button>
                        <Button onClick={handleCreateAnnOk}>OK</Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>
            <Modal isOpen={isHyperparametersModalOpen} onClose={toggleHyperparametersModal}>
                <ModalOverlay />
                <ModalContent>
                    <ModalHeader>PSO Hyperparameters</ModalHeader>
                    <ModalCloseButton />
                    <ModalBody>
                        <VStack spacing={6}>
                            <HStack spacing={4}>
                                <FormControl>
                                    <FormLabel>Number of Particles</FormLabel>
                                    <NumberInput value={numOfParticles} onChange={setNumOfParticles}>
                                    <NumberInputField />
                                    </NumberInput>
                                </FormControl>
                                <FormControl>
                                    <FormLabel>Alpha</FormLabel>
                                    <NumberInput value={inertiaWeight} onChange={setInertiaWeight}>
                                    <NumberInputField />
                                    </NumberInput>
                                </FormControl>
                            </HStack>
                            <HStack spacing={4}>
                                <FormControl>
                                    <FormLabel>Beta</FormLabel>
                                    <NumberInput value={cognitiveWeight} onChange={setCognitiveWeight} min={0} max={1}>
                                        <NumberInputField />
                                    </NumberInput>
                                </FormControl>
                                <FormControl>
                                    <FormLabel>Gamma</FormLabel>
                                    <NumberInput value={socialWeight} onChange={setSocialWeight} min={0} max={1}>
                                        <NumberInputField />
                                    </NumberInput>
                                </FormControl>
                            </HStack>
                            <HStack spacing={4}>
                                <FormControl>
                                    <FormLabel>Delta</FormLabel>
                                    <NumberInput value={globalWeight} onChange={setGlobalWeight} min={0} max={1}>
                                        <NumberInputField />
                                    </NumberInput>
                                </FormControl>
                                <FormControl>
                                    <FormLabel>Number of Iterations</FormLabel>
                                    <NumberInput value={numIterations} onChange={setNumIterations}>
                                        <NumberInputField />
                                    </NumberInput>
                                </FormControl>
                            </HStack>
                            <FormControl>
                                <FormLabel>Number of Informants</FormLabel>
                                <NumberInput value={numInformants} onChange={setNumInformants}>
                                    <NumberInputField />
                                </NumberInput>
                            </FormControl>
                            <FormControl>
                                <FormLabel>Inertia Weights Range</FormLabel>
                                <HStack>
                                    <NumberInput value={r1Range.min} onChange={value => setR1Range({ ...r1Range, min: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                    <NumberInput value={r1Range.max} onChange={value => setR1Range({ ...r1Range, max: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                </HStack>
                            </FormControl>
                            <FormControl>
                                <FormLabel>R1 Range</FormLabel>
                                <HStack>
                                    <NumberInput value={r1Range.min} onChange={value => setR1Range({ ...r1Range, min: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                    <NumberInput value={r1Range.max} onChange={value => setR1Range({ ...r1Range, max: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                </HStack>
                            </FormControl>
                            <FormControl>
                                <FormLabel>R2 Range</FormLabel>
                                <HStack>
                                    <NumberInput value={r2Range.min} onChange={value => setR2Range({ ...r2Range, min: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                    <NumberInput value={r2Range.max} onChange={value => setR2Range({ ...r2Range, max: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                </HStack>
                            </FormControl>
                            <FormControl>
                                <FormLabel>R3 Range</FormLabel>
                                <HStack>
                                    <NumberInput value={r3Range.min} onChange={value => setR3Range({ ...r3Range, min: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                    <NumberInput value={r3Range.max} onChange={value => setR3Range({ ...r3Range, max: value })}>
                                        <NumberInputField />
                                    </NumberInput>
                                </HStack>
                            </FormControl>
                        </VStack>
                    </ModalBody>
                    <ModalFooter>
                        <Button colorScheme="blue" mr={3} onClick={toggleHyperparametersModal}>Cancel</Button>
                        <Button onClick={handlePSOSubmit}>OK</Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>
        </>
    );
}

export default SideMenu;