// ignore_for_file: prefer_const_constructors
import 'dart:io';

import 'package:equationsolver/ui/components/elevatedbutton.dart';
import 'package:equationsolver/ui/finalScreen.dart';
import 'package:flutter/material.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
            gradient: LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: const [Colors.black, Color.fromARGB(255, 61, 61, 64)])),
        child: Center(
          child: MyElevatedButton(
            onPressed: () {
              Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => FinalScreen(),
                  ));
            },
            borderRadius: BorderRadius.circular(15),
            child: Text(
              'Scan The Equation',
              style: TextStyle(fontSize: 26),
            ),
          ),
        ),
      ),
    );
  }
}
