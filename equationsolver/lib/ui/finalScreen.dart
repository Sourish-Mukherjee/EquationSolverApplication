import 'package:flutter/material.dart';

class FinalScreen extends StatefulWidget {
  const FinalScreen({Key? key}) : super(key: key);

  @override
  State<FinalScreen> createState() => _FinalScreenState();
}

class _FinalScreenState extends State<FinalScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
            gradient: LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: const [Colors.black, Color.fromARGB(255, 61, 61, 64)])),
        child: Container(
          padding: const EdgeInsets.only(left: 5),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Flexible(
                flex: 1,
                child: Container(
                  padding: const EdgeInsets.only(left: 10, top: 30),
                  alignment: Alignment.topLeft,
                  child: const Text(
                    "Detected Equation",
                    style: TextStyle(
                        fontSize: 20,
                        color: Color.fromARGB(255, 255, 255, 255)),
                  ),
                ),
              ),
              const Flexible(
                flex: 1,
                child: Padding(
                  padding: EdgeInsets.all(15.0),
                  child: Divider(
                      height: 3,
                      thickness: 2,
                      color: Color.fromARGB(255, 6, 104, 117)),
                ),
              ),
              Flexible(
                flex: 2,
                child: Container(
                  padding: const EdgeInsets.only(left: 10, top: 15),
                  child: const Text(
                    "Result",
                    style: TextStyle(
                        fontSize: 20,
                        color: Color.fromARGB(255, 255, 255, 255)),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
