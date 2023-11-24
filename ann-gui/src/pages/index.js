import { useState } from 'react';
import { Text, Flex, Button, Checkbox, FormControl, FormLabel, Input, VStack, Modal, ModalOverlay, ModalContent, ModalHeader, ModalFooter, ModalBody, ModalCloseButton, NumberInput, NumberInputField, NumberInputStepper, NumberIncrementStepper, NumberDecrementStepper, Heading, Divider, Spacer, HStack, Box } from "@chakra-ui/react";
import { CheckIcon } from "@chakra-ui/icons";
import axios from 'axios';
import SideMenu from './sideMenu';

export default function Home() {
  const [isOpen, setIsOpen] = useState(true);
  const [normalize, setNormalize] = useState(false);
  const [skipHeader, setSkipHeader] = useState(false);
  const [testSplit, setTestSplit] = useState(0.2);
  const [file, setFile] = useState(null);
  const [responseData, setResponseData] = useState(null);
  const [inputLayerNodes, setInputLayerNodes] = useState();
  const [neuralNetworkPlot, setNeuralNetworkPlot] = useState('');
  const [annCreated, setAnnCreated] = useState(false);
  const [annDetails, setAnnDetails] = useState({});


  const toggleModal = () => {
    setIsOpen(!isOpen);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleOk = async () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('normalize', normalize);
    formData.append('skipHeader', skipHeader);
    formData.append('testSplit', testSplit);

    const response = await axios.post('http://localhost:5000/saveData', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  
    try {
      const response = await axios.post('http://localhost:5000/saveData', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      console.log("API Response:", response.data);

      if (response.data && response.data.xShape && response.data.xShape.length > 1) {
          setResponseData(response.data);
          setInputLayerNodes(response.data.xShape[1]);
      } else {
          console.error("Invalid response structure");
      }
      } catch (error) {
          console.error("Error in API call", error);
      }

      setIsOpen(false);  
  };

  const checkInputLayerNodes = () => {
    if (inputLayerNodes) {
        console.log("Input Layer Nodes:", inputLayerNodes);
        // Additional logic if needed
    } else {
        console.error("Input Layer Nodes not set");
    }
  };

  return (
    <Box
      bgImage="url(./bg.jpg)"
      bgPosition="center"
      bgRepeat="no-repeat"
      bgSize="cover"
      w="100%"
      h="100vh"
      p={4}
    >
        <Flex direction="row" align="flex-start" justify="space-between" h={"100%"}>
          <HStack spacing={4} w="10%" h={"60%"}>
          <SideMenu 
            onBrowseData={toggleModal} 
            checkInputLayerNodes={checkInputLayerNodes} 
            setNeuralNetworkPlot={setNeuralNetworkPlot} 
            setAnnCreated={setAnnCreated} 
            setAnnDetails={setAnnDetails} 
            setResponseData={setResponseData}
          />
          </HStack>
          {/* Information Window 1 */}
          <Box p={4} borderWidth="2px" borderRadius="lg" w="50%" backgroundColor={"whiteAlpha.900"} h={"60%"} borderStyle="solid" borderColor="black">
            {responseData && (
                <VStack align="start" spacing={2}>
                <Heading size="lg" color="teal.500">Dataset Information</Heading>
                <Divider my={3} borderColor="teal.300" />
              
                <Text fontSize="md" fontWeight="bold">
                  File: <Text as="span" fontWeight="normal">{responseData.fileName ?? "N/A"}</Text>
                </Text>
              
                <Text fontSize="md" fontWeight="bold">
                  Shape: <Text as="span" fontWeight="normal">{responseData.shape ? responseData.shape.toString() : "N/A"}</Text>
                </Text>
              
                <Text fontSize="md" fontWeight="bold">
                  X shape: <Text as="span" fontWeight="normal">{responseData.xShape ? responseData.xShape.toString() : "N/A"}</Text>, Y shape: <Text as="span" fontWeight="normal">{responseData.yShape ? responseData.yShape.toString() : "N/A"}</Text>
                </Text>
              
                <Text fontSize="md" fontWeight="bold">
                  Normalization: <Text as="span" fontWeight="normal">{responseData.normalize !== undefined ? responseData.normalize.toString() : "N/A"}</Text>
                </Text>
              
                <Text fontSize="md" fontWeight="bold">
                  Split Ratio: <Text as="span" fontWeight="normal">{responseData.splitRatio ?? "N/A"}</Text>
                </Text>
              </VStack>            
            )}
            {annCreated && (
                <VStack align="start" spacing={2}>
                <Heading size="lg" color="teal.500">Artificial Neural Network (ANN) Configuration</Heading>
                <Divider my={3} borderColor="teal.300" />
                <Text fontSize="md">
                  <strong>Input Layer:</strong> {annDetails.input_size} nodes
                </Text>
                {annDetails.hidden_layers.map((layer, index) => (
                  <Text key={index} fontSize="md">
                    <strong>Hidden Layer {index + 1}:</strong> {layer.neurons} neurons, Activation Function: {layer.activation}
                  </Text>
                ))}
                <Text fontSize="md">
                  <strong>Output Layer:</strong> {annDetails.output_layer.neurons} nodes, Activation Function: {annDetails.output_layer.activation}
                </Text>
                <Text fontSize="md">
                  <strong>Loss Function:</strong> {annDetails.loss_function}
                </Text>
              </VStack>                        
            )}
          </Box>

          {/* Information Window 2 */}
          <Box p={4} borderWidth="2px" borderRadius="lg" w="30%" backgroundColor={"whiteAlpha.900"} h={"60%"} borderStyle="solid" borderColor="black" alignContent={"center"} justifyContent={"center"}>
            {neuralNetworkPlot && (
                <img 
                    src={`data:image/png;base64,${neuralNetworkPlot}`} 
                    alt="Neural Network Visualization" 
                    style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                />
            )}
          </Box>
        </Flex>
      <VStack spacing={4} padding={4} alignItems='flex-start'>
        <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} size="xl">
          <ModalOverlay />
          <ModalContent padding={5} borderRadius="xl" boxShadow="2xl">
            <ModalHeader>
              <Heading size="lg" color="teal.500">Load the Dataset in CSV</Heading>
              <Divider my={3} borderColor="teal.300" />
            </ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <VStack spacing={5}>
                <FormControl>
                  <Input type="file" size="lg" variant="flushed" focusBorderColor="teal.400" onChange={handleFileChange}/>
                </FormControl>
                <VStack spacing={2} alignItems="flex-start">
                  <Checkbox size="lg" isChecked={normalize} onChange={(e) => setNormalize(e.target.checked)} colorScheme="teal" iconColor="white" icon={<CheckIcon boxSize="1.5em" />}>
                    Normalize
                  </Checkbox>
                  <Spacer />
                  <Checkbox size="lg" isChecked={skipHeader} onChange={(e) => setSkipHeader(e.target.checked)} colorScheme="teal" iconColor="white" icon={<CheckIcon boxSize="1.5em" />}>
                    Skip Header
                  </Checkbox>
                </VStack>
                <FormControl>
                  <FormLabel> Test Split</FormLabel>
                  <NumberInput size="lg" value={testSplit} onChange={(value) => setTestSplit(value)} focusBorderColor="teal.400" min={0.0} max={0.9} step={0.1} precision={1}>
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>
              </VStack>
            </ModalBody>
            <ModalFooter>
              <Button colorScheme="teal" size="lg" onClick={handleOk}>
                Ok
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </VStack>
    </Box>
  );
}
